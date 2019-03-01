# Pattern Matching / Word Frequency in CUDA

Pattern matching in textual chunks involves looking for certain keywords in a
long stream of text. This could have lots of applications in the humanities and
social sciences, and is also possibly relevant to some bioinformatics applications
such as gene sequencing. Other applications which require fast pattern matching
are antivirus engines, web search engines, text editors etc.  

CPU code implements naive pattern
matching based on a fixed number of keywords. In our case, we match pattern
in a long text stream with 32 keywords. These keywords have 4 characters each 
which are held in a single 32-bit unsigned integer so that a single integer com-
parison checks for a match with a corresponding integer holding 4 characters of the input text. 
Because the beginning of each word in the text is not necessarily
aligned with the beginning of an integer, the CPU code has to also consider 1, 2 and 3-byte offsets.  

At the end of matching all the keywords, you need to report frequency of
all the keywords in the given text. You are given 4 text files of different sizes
in ”data/ ” folder and this folder also contains the 32 keywords. Understand
the given CPU code to generate the keyword frequencies from the given texts
in ”data/ ” folder.

## Usage

```
make all
./pattern_tiling
./pattern_streams
```
## Strategy

### Titling

`NWORDS` -> number of keywords we need to find.  

Each block is 1-D with the size of the number of keywords that we need to find. This means that `BLOCK_DIM = (NWORDS,1,1)`. Each thread in the block is responsible for finding the number of the number of occurrences of one keyword(i.e assigned to the thread) in the text loaded into the shared memory of that block. In the end, the occurrence number is updated atomically to global memory. Each thread loads text from global memory using titling. That’s each thread loads `TILE_SIZE+TILE_SIZE`(to handle corner cases) elements from the global memory of text.  

- Launch kernel with following config:
    - Grid Size: (`Length Of Text`)/(`TILE_SIZE*NWORDS`)
    - Block Size: `NWORDS`
- load `text` into shared memory using titling
- assign one keyword to each thread in a block
- iterate through all memory location in the shared memory corresponding to the block to find the word count in that block. In other words, each thread will go through `TILE_SIZE * NWORDS` words to match a keyword and update a local sum counter
- Atomically update add the sum counter to a global array

### Streams

We use a 8-way stream with `TILE_SIZE=1`.  


## Report

Tiling: ~300x speedup (including memcopy)  
Stream: ~500x speedup (including memcopy)  
