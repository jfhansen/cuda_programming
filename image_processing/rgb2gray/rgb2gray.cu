/**************************************************************
 * File: rgb2gray.cu
 * Description: CUDA implementation of application that transfers
 * color picture to grayscale.
 *
 * Author: jfhansen
 * Last Modification: 28/07/2020
 *************************************************************/

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKDIM 16

/* Converts RGBA image to Grayscale image
 * When converting image from RGB to grayscale photo,
 * the pixels should use the following proportion of red, green and blue:
 * I = 0.299f * R + 0.587f * G + 0.114f * B
 * Arguments: 
 * rgbaImage: constant pointer to  array of uchar4 holding RGBA values.
 * grayImage: pointer to array of chars.
 * numrows, numcols: Number of pixel rows and columns */
__global__ void cuda_rgba_to_grayscale(const uchar4 *const rgbaImage,
	unsigned char *const grayImage, int numRows, int numCols)
{
	// Get row and column for pixel
	unsigned col, row;
	col = threadIdx.x + blockDim.x * blockIdx.x;
	row = threadIdx.y + blockDim.y * blockIdx.y;
	// Fetch rgba value at pixel
	uchar4 pixel = rgbaImage[row*numCols+col];
	unsigned char brightness = (unsigned char)(.299f * pixel.x + .587f * pixel.y + .114f * pixel.z);
	// Compute pixel brightness
	grayImage[row*numCols+col] = brightness;
}

// Transfers h_rgbaImage to device, converts RGBA image to grayscale and transfers
// resulting grayscale image to host memory, h_grayImage.
void rgba_to_grayscale(const uchar4 *const d_rgbaImage,	unsigned char *const d_grayImage, 
						size_t numRows, size_t numCols)
{
	dim3 threadsPerBlock(BLOCKDIM,BLOCKDIM,1);
	dim3 blocksPerGrid(
		(numCols + BLOCKDIM - 1)/BLOCKDIM,
		(numRows + BLOCKDIM - 1)/BLOCKDIM, 
		1);

	cuda_rgba_to_grayscale<<<blocksPerGrid, threadsPerBlock>>>(d_rgbaImage, d_grayImage, numRows, numCols);

	cudaError_t err;
	while ( (err = cudaGetLastError()) != cudaSuccess )
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}


