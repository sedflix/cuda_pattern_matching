/*****************************************************************************
*
* String Pattern Matching - Serial Implementation
* 
* Reference: http://people.maths.ox.ac.uk/~gilesm/cuda/
*
*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

// Includes CUDA
#include <cuda_runtime.h>

#define LINEWIDTH 20

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

__global__ void matchPattern_gpu_1(unsigned int *text, unsigned int *words, int *matches, int nwords, int length)
{
	unsigned int word;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < length)
    {
		for (int offset=0; offset<4; offset++)
		{
			if (offset==0)
				word = text[idx];
			else
				word = (text[idx]>>(8*offset)) + (text[idx+1]<<(32-8*offset)); 

			for (int w=0; w<nwords; w++){
				if (word==words[w]){
					atomicAdd(&matches[w],1);
				}
			} 	
		}
	}
}


int main(int argc, const char **argv)
{

	int length, len, nwords=5, matches[nwords];
	char *ctext, keywords[nwords][LINEWIDTH], *line;
	line = (char*) malloc(sizeof(char)*LINEWIDTH);
	unsigned int  *text,  *words;
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



	fp = fopen("./data/small.txt","r");
	if (!fp)
	{	printf("Unable to open the file.\n");	exit(0);}

	length = 0;
	while (getc(fp) != EOF) length++;
	ctext = (char *) malloc(length+4);

	rewind(fp);

	for (int l=0; l<length; l++) ctext[l] = getc(fp);
	for (int l=length; l<length+4; l++) ctext[l] = ' ';

	fclose(fp);

	printf("Length : %d\n", length );
	// define number of words of text, and set pointers
	len  = length/4;
	text = (unsigned int *) ctext;

	// define words for matching
	words = (unsigned int *) malloc(nwords*sizeof(unsigned int));

	for (int w=0; w<nwords; w++)
	{
		words[w] = ((unsigned int) keywords[w][0])
             + ((unsigned int) keywords[w][1])*(1<<8)
             + ((unsigned int) keywords[w][2])*(1<<16)
             + ((unsigned int) keywords[w][3])*(1<<24);

	}
	// CPU execution
	const clock_t begin_time = clock();
	matchPattern_CPU(text, words, matches, nwords, len);
	float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;
	printf("Time for matching keywords: %fs\n\n", runTime);

	printf("Printing Matches:\n");
	printf("Word\t  |\tNumber of Matches\n===================================\n");
	for (int i = 0; i < nwords; ++i)
		printf("%s\t  |\t%d\n", keywords[i], matches[i]);


	// GPU execution
	unsigned int *d_text; unsigned int *d_words; int *d_matches;
	int *h_matches;
	h_matches = (int *)malloc(nwords*sizeof(int));

	cudaMalloc((void**)&d_words, nwords*sizeof(unsigned int));
	cudaMalloc((void**)&d_matches, nwords*sizeof(int));
	cudaMalloc((void**)&d_text, sizeof(char)*strlen(ctext));


	cudaMemcpy(d_text, text, sizeof(char)*strlen(ctext), cudaMemcpyHostToDevice);
	cudaMemcpy(d_words, words, nwords*sizeof(unsigned int), cudaMemcpyHostToDevice);

	matchPattern_gpu_1<<<len/32,32>>>(d_text, d_words, d_matches, nwords, len);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaMemcpy(h_matches, d_matches, nwords*sizeof(int), cudaMemcpyDeviceToHost));

	for(int i = 0; i<nwords; i++) {
		if(matches[i] != h_matches[i]) {
			printf("WRONG OUTPUT:\t %s\t|\t%d\n",  keywords[i], h_matches[i]);
		}
	}
	
	free(ctext);
	free(words);
	cudaFree(d_words);
	cudaFree(d_matches);
	cudaFree(d_text);
	
}
