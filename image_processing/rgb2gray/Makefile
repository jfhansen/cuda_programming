OCV = `pkg-config --cflags --libs opencv`
CCAP = -gencode arch=compute_30,code=compute_30
CU = -I /usr/local/cuda-11.0/samples/common/inc

main : im_processing.o rgb2gray.o
	nvcc main.cpp im_processing.o rgb2gray.o ${OCV} -o main.out

im_processing.o : ../im_processing.cpp
	nvcc -c ../im_processing.cpp

rgb2gray.o : rgb2gray.cu
	nvcc -c rgb2gray.cu ${CCAP} ${CU}

clean :
	rm -rf *.o
	rm -rf main.out