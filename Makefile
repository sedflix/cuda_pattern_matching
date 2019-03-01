
CPPCOMP := g++
CPPFLAGS := -Wno-deprecated-declarations

CUCOMP := nvcc
CUFLAGS := --gpu-architecture=compute_60  -Wno-deprecated-declarations

all:
	${CUCOMP} pattern.cu -o pattern ${CUFLAGS}

clean:
	rm -rf pattern
