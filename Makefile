
CPPCOMP := g++
CPPFLAGS := -Wno-deprecated-declarations

CUCOMP := nvcc
CUFLAGS := --gpu-architecture=compute_60  -Wno-deprecated-declarations

all: tiling streams
	echo ""
tiling:
	${CUCOMP} pattern_tiling.cu -o pattern_tiling ${CUFLAGS}
streams:
	${CUCOMP} pattern_streams.cu -o pattern_streams ${CUFLAGS}

clean:
	rm -rf pattern_titling pattern_streams
