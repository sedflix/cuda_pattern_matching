#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
// Includes CUDA
#include <cuda_runtime.h>

#define LINEWIDTH 20
#define NWORDS 32
#define N_STREAMS 8
#define BLOCK_SIZE 32
#define TITLE_SIZE 1

int length;
int len;
int nwords;
int matches[NWORDS];
char *ctext;
char keywords[NWORDS][LINEWIDTH];
unsigned int  *text;
unsigned int  *words;
float cpuRunTime;

// citation: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void intialise(char *input)
{
    nwords = NWORDS;
    printf("-----------\nGoint to read %s\n", input);
    char *line;
	line = (char*) malloc(sizeof(char)*LINEWIDTH);
	memset(matches, 0, sizeof(matches));


	// read in text and keywords for processing
	FILE *fp, *wfile;
	wfile = fopen("./data/keywords.txt","r");
	if (!wfile)
	{	printf("keywords.txt: File not found.\n");	exit(0);}

	int k=0, cnt = nwords;
	size_t read, linelen = LINEWIDTH;
	while((read = getline(&line, &linelen, wfile)) != -1 && cnt--)
	{
		strncpy(keywords[k], line, sizeof(line));
		keywords[k][4] = '\0';
		k++;
	}
	fclose(wfile);


	fp = fopen(input,"r");
	if (!fp)
	{	printf("Unable to open the file.\n");	exit(0);}

	length = 0;
	while (getc(fp) != EOF) length++;
	ctext = (char *)malloc(length+4);

	rewind(fp);

	for (int l=0; l<length; l++) ctext[l] = getc(fp);
	for (int l=length; l<length+4; l++) ctext[l] = ' ';

	fclose(fp);

	printf("Length : %d\n", length );
	// define number of words of text, and set pointers
	len  = length/4;
	text = (unsigned int *) ctext;

	// define words for matching
	words = (unsigned int *)malloc(nwords*sizeof(unsigned int));

	for (int w=0; w<nwords; w++)
	{
		words[w] = ((unsigned int) keywords[w][0])
             + ((unsigned int) keywords[w][1])*(1<<8)
             + ((unsigned int) keywords[w][2])*(1<<16)
             + ((unsigned int) keywords[w][3])*(1<<24);

	}
}

void deinit(){
	free(words);
    free(text);
}

void check_matches(int *temp_matches){
	bool isRight = true;
    for(int i = 0; i<NWORDS; i++) {
		if(matches[i] != temp_matches[i]) {
			isRight = false;
			printf("WRONG OUTPUT:\t %s\t|\t%d\n",  keywords[i], temp_matches[i]);
        }
	}

	if(isRight) {
		printf(" - Correct Answer -\n");
	}
}

void print_matches(int *temp_matches){
    printf("Printing Matches:\n");
	printf("Word\t  |\tNumber of Matches\n===================================\n");
	for (int i = 0; i < nwords; ++i)
		printf("%s\t  |\t%d\n", keywords[i], temp_matches[i]);

}

void matchPattern_CPU(unsigned int *text, unsigned int *words, int *matches, int nwords, int length)
{
	unsigned int word;

	for (int l=0; l<length; l++)
	{
		for (int offset=0; offset<4; offset++)
		{
			if (offset==0)
				word = text[l];
			else
				word = (text[l]>>(8*offset)) + (text[l+1]<<(32-8*offset)); 

			for (int w=0; w<nwords; w++){
				matches[w] += (word==words[w]);
			} 
				
		}
	}
}

void exec_CPU(){
    // CPU execution
	const clock_t begin_time = clock();
	matchPattern_CPU(text, words, matches, nwords, len);
	cpuRunTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	printf("CPU exec time: %f s\n\n", cpuRunTime);
}

__global__ void matchPattern_gpu_1(unsigned int *text, unsigned int *words, int *matches, int nwords, int length, int offset_, int which)
{
	int tid = threadIdx.x;
	int idx = offset_ + blockIdx.x * blockDim.x + tid;
	
	// for loading text into the shared memory
	__shared__ unsigned int text_s[NWORDS + 1];
	text_s[tid] = text[idx];
	text_s[NWORDS] = text[offset_ + (blockIdx.x * blockDim.x) + blockDim.x];

	// loads the keyword for this thread
	// each thread in a block is reponsible for one keyword
	unsigned int keyword = words[tid];
	
	__syncthreads();
	
	unsigned int word;
	int sum = 0;
 

		
	#pragma loop unroll
	for(int w = 0; w < NWORDS; w++) {
		#pragma loop unroll
		for (int offset=0; offset<4; offset++)
		{
			word = offset==0 ? text_s[w] : (text_s[w]>>(8*offset)) + (text_s[w+1]<<(32-8*offset)); 
			sum = sum + (word==keyword);
		}
	}

	atomicAdd(&matches[(which*NWORDS)+tid],sum);
}

// citation: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
void exec_gpu_stream(){

	const int nStreams = N_STREAMS;
	const int streamSize = len / nStreams;
	const int straming_bytes = streamSize * sizeof(unsigned int);

	unsigned int *d_text; unsigned int *d_words; int *d_matches;
	int *h_matches;
	h_matches = (int *)malloc(nwords*sizeof(int)*N_STREAMS);
	memset(h_matches, 0, nwords*sizeof(int)*N_STREAMS);

	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; i++){
		checkCudaErrors( cudaStreamCreate(&stream[i]));
	}

	cudaHostRegister(words,nwords*sizeof(int),0);
	cudaHostRegister(text,strlen(ctext)*sizeof(char),0);
	cudaHostRegister(h_matches,nwords*sizeof(int),0);

	checkCudaErrors(cudaMalloc((void**)&d_words, nwords*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_matches, nwords*sizeof(int)*nStreams));
	checkCudaErrors(cudaMalloc((void**)&d_text, sizeof(unsigned int)*len));
		
	cudaEvent_t start, stop;	
	float tiime_ = 0;
	checkCudaErrors( cudaEventCreate(&start) );
	checkCudaErrors( cudaEventCreate(&stop) );

	checkCudaErrors( cudaEventRecord(start,0) );
	checkCudaErrors(cudaMemcpy(d_words, words, nwords*sizeof(unsigned int), cudaMemcpyHostToDevice));
	for (int i = 0; i < nStreams; ++i) 
	{
		int offset = i * streamSize;
		checkCudaErrors(cudaMemcpyAsync(&d_text[offset], &text[offset], straming_bytes, cudaMemcpyHostToDevice, stream[i]));
	}
	
	for (int i = 0; i < nStreams; ++i) 
	{
		int offset = i * streamSize;
		matchPattern_gpu_1<<<ceil(streamSize/(TITLE_SIZE*NWORDS)), NWORDS, 0, stream[i]>>>(d_text, d_words, d_matches, nwords, len, offset, i);
	}

	for (int i = 0; i < nStreams; ++i) 
	{
		int offset = i * streamSize;
		checkCudaErrors(cudaMemcpyAsync(&h_matches[(i*NWORDS)], &d_matches[(i*NWORDS)], NWORDS*sizeof(int), cudaMemcpyDeviceToHost, stream[i]));
	}
	// cudaMemcpy(h_matches, d_matches, nwords*sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaErrors( cudaEventRecord(stop, 0) );
	checkCudaErrors( cudaEventSynchronize(stop) );
	checkCudaErrors( cudaEventElapsedTime(&tiime_, start, stop) );
	  
	printf("Time kernel+memory: %fs\n", tiime_/1000);
	printf("Speedup with memory: %f\n", cpuRunTime/((tiime_)/1000));

	for(int w=0; w < NWORDS; w++)
	{
		for (int i = 1; i < nStreams; ++i)
		{
			h_matches[w] += h_matches[(i*NWORDS) + w];
		}
	}

	check_matches(h_matches);

	// cleanup
	checkCudaErrors( cudaEventDestroy(start) );
	checkCudaErrors( cudaEventDestroy(stop) );

	for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors( cudaStreamDestroy(stream[i]));
	}
	
	cudaHostUnregister(text);
	cudaHostUnregister(words);
	cudaHostUnregister(h_matches);

	cudaFree(d_words);
	cudaFree(d_matches);
	cudaFree(d_text);
	
}
void exec_gpu_simple(){

		// GPU execution
		unsigned int *d_text; unsigned int *d_words; int *d_matches;
		int *h_matches;
		h_matches = (int *)malloc(nwords*sizeof(int));


		checkCudaErrors(cudaMalloc((void**)&d_words, nwords*sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**)&d_matches, nwords*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_text, sizeof(unsigned int)*len));

		cudaEvent_t start,stop;
		float time_H2D,time_D2H,time_kernel;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));


		// MEMCOPY
		cudaEventRecord(start, 0);
		checkCudaErrors(cudaMemcpy(d_words, words, nwords*sizeof(unsigned int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_text, text, sizeof(unsigned int)*len, cudaMemcpyHostToDevice));
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_H2D,start,stop);
		printf("HostToDevice memcopy time: %fs\n", time_H2D/1000);

		// RUN KERNEL
		cudaEventRecord(start, 0);
		matchPattern_gpu_1<<< ceil((float)len/(TITLE_SIZE*NWORDS)),NWORDS>>>(d_text, d_words, d_matches, nwords, len, 0, 0);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		checkCudaErrors(cudaPeekAtLastError());
		cudaEventElapsedTime(&time_kernel,start,stop);
		printf("Kernel execution time: %fs\n", time_kernel/1000);
		
		cudaEventRecord(start, 0);
		checkCudaErrors(cudaMemcpy(h_matches, d_matches, nwords*sizeof(int), cudaMemcpyDeviceToHost));
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_D2H,start,stop);
		printf("DeviceToHost memcopy time: %fs\n", time_D2H/1000);

		printf("Total memcopy time: %fs\n", (time_D2H+time_H2D)/1000);
		printf("Total memcopy+kernel time: %fs\n", (time_D2H+time_H2D+time_kernel)/1000);
		
		printf("Speedup without memory: %f\n", cpuRunTime/((time_kernel)/1000));
		printf("Speedup with memory: %f\n", cpuRunTime/((time_D2H + time_H2D + time_kernel)/1000));
		
		check_matches(h_matches);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

	    free(h_matches);
		cudaFree(d_words);
		cudaFree(d_matches);
		cudaFree(d_text);
}

int main(int argc, const char **argv)
{

	intialise("./data/small.txt");
    exec_CPU();
	exec_gpu_stream();

	intialise("./data/medium.txt");
    exec_CPU();
	exec_gpu_stream();

    intialise("./data/large.txt");
    exec_CPU();
	exec_gpu_stream();

	deinit();
	
}
