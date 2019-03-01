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

## Report

Tiling: ~300x speedup (including memcopy)  
Stream: ~500x speedup (including memcopy)  
