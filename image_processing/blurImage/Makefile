OCV = `pkg-config --cflags --libs opencv`
CCAP = -gencode arch=compute_30,code=compute_30
CU_INC = -I /usr/local/cuda-11.0/samples/common/inc

main : im_processing.o blurImage.o
	nvcc main.cpp im_processing.o blurImage.o ${OCV} -o main.out

im_processing.o : im_processing.cpp
	nvcc -c im_processing.cpp ${CCAP} ${CU_INC}

blurImage.o : blurImage.cu
	nvcc -c blurImage.cu ${CCAP} ${CU_INC}

clean :
	rm -rf *.o
	rm -rf main.out