/************************************************************** 
 * File: blurImage.cu 
 * Description: CUDA implementation of application that blurs
 * images using stencil operation. Helper functions separate
 * color image to R, G and B channels, as well as combine the
 * R G & B channels to a color image. 
 * 
 * Author: jfhansen 
 * Last Modification: 30/07/2020 
 *************************************************************/ 
 
#include <iostream> 
#include <stdio.h> 
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <helper_functions.h>
#include <helper_cuda.h>
 
#define BLOCKDIM 16 
__constant__ float d_filter[9*9];

__global__ void cuda_gaussian_blur(const unsigned char *const inputChannel,
							  unsigned char *const outputChannel, 
							  size_t numRows, size_t numCols,
							  const float* const filter, const int filterWidth)
{
	// Get 2D position of thread in image
	const int2 thread_2D_pos = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
										 threadIdx.y + blockDim.y * blockIdx.y);
	// Get 1D position (row-major) of thread in image
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.y >= numRows || thread_2D_pos.x >= numCols)
		return;

	int image_r, image_c;
	float filter_value, image_value, result = 0.f;
	for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; filter_r++)
	{
		for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; filter_c++)
		{
			image_r = min(max(thread_2D_pos.y + filter_r, 0), static_cast<int>(numRows-1));
			image_c = min(max(thread_2D_pos.x + filter_c, 0), static_cast<int>(numCols-1));
			
			image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
			filter_value = d_filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

			result += image_value * filter_value;
		}
	}
	outputChannel[thread_1D_pos] = static_cast<unsigned char>(result);
}

__global__ void cuda_separate_channels(const uchar4 *const rgbaImage,
	size_t numRows, size_t numCols, unsigned char *const redChannel, 
	unsigned char *const greenChannel,unsigned char *const blueChannel)
{
	// Get 2D position of thread in image
	const int2 thread_2D_pos = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
										 threadIdx.y + blockDim.y * blockIdx.y);
	// Get 1D position (row-major) of thread in image
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	
	// Check that thread position is within bounds of image
	if (thread_2D_pos.y >= numRows || thread_2D_pos.x >= numCols)
		return;
	
	// Extract pixel from rgbaImage
	uchar4 pixel = rgbaImage[thread_1D_pos];
	// Write R, G and B values to corresponding channel arrays
	redChannel[thread_1D_pos] = pixel.x;
	greenChannel[thread_1D_pos] = pixel.y;
	blueChannel[thread_1D_pos] = pixel.z;
}

__global__ void cuda_combine_channels(const unsigned char *const redChannel,
	const unsigned char *const greenChannel, const unsigned char *const blueChannel,
	size_t numRows, size_t numCols, uchar4 *const rgbaImage)
{
	// Get 2D thread position
	const int2 thread_2D_pos = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
										 threadIdx.y + blockDim.y * blockIdx.y);
	// Get 1D thread position
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	
	// Check that thread position is within bounds of image
	if (thread_2D_pos.y >= numRows || thread_2D_pos.x >= numCols)
		return;
	
	// Extract red, green and blue channel values for pixel
	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];
	// alpha = 255, for no transparency
	uchar4 pixel = make_uchar4(red,green,blue,255);
	// Write pixel to rgbaImage
	rgbaImage[thread_1D_pos] = pixel;
}

unsigned char *d_red, *d_green, *d_blue, *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
//float *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
								const float* const h_filter, const size_t filterWidth)
{
	unsigned CHANNEL_BYTES = sizeof(unsigned char)*numRowsImage*numColsImage;
	unsigned FILTER_BYTES = sizeof(float)*filterWidth*filterWidth;
	// Allocate memory for channels
	checkCudaErrors(cudaMalloc(&d_red, CHANNEL_BYTES));
	checkCudaErrors(cudaMalloc(&d_green, CHANNEL_BYTES));
	checkCudaErrors(cudaMalloc(&d_blue, CHANNEL_BYTES));
	checkCudaErrors(cudaMalloc(&d_redBlurred, CHANNEL_BYTES));
	checkCudaErrors(cudaMalloc(&d_greenBlurred, CHANNEL_BYTES));
	checkCudaErrors(cudaMalloc(&d_blueBlurred, CHANNEL_BYTES));

	// Allocate memory for the filter
	//checkCudaErrors(cudaMalloc(&d_filter, FILTER_BYTES));
	//cudaMalloc(&d_filter, FILTER_BYTES);

	// Copy filter on host to device
	//checkCudaErrors(cudaMemcpy(d_filter, h_filter, FILTER_BYTES, cudaMemcpyHostToDevice));
	cudaMemcpyToSymbol(d_filter, h_filter, FILTER_BYTES);//, cudaMemcpyHostToDevice);
	cudaError_t err;
	while ( (err = cudaGetLastError()) != cudaSuccess)
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void gaussian_blur(const uchar4 *const d_inImage, uchar4 *const d_outImage,
	size_t numRows, size_t numCols, const size_t filterWidth)
{
	// Compute block and grid dimensions
	dim3 threads(BLOCKDIM, BLOCKDIM, 1);
	dim3 blocks( (numCols + BLOCKDIM - 1) / BLOCKDIM,
				 (numRows + BLOCKDIM - 1) / BLOCKDIM, 1);
	// Launch channel separation kernel
	cuda_separate_channels<<<blocks, threads>>>(d_inImage, numRows, numCols,
		d_red, d_green, d_blue);
	// Synchronize
	cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

	// Launch gaussian blur kernel on the red channel
	cuda_gaussian_blur<<<blocks, threads>>>(d_red, d_redBlurred, numRows, numCols,
											d_filter, filterWidth);
	// Launch gaussian blur kernel on the green channel
	cuda_gaussian_blur<<<blocks, threads>>>(d_green, d_greenBlurred, numRows, numCols,
											d_filter, filterWidth);
	// Launch gaussian blur kernel on blue channel
	cuda_gaussian_blur<<<blocks, threads>>>(d_blue, d_blueBlurred, numRows, numCols,
											d_filter, filterWidth);
	// Synchronize
	cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

	// Recombine channels
	cuda_combine_channels<<<blocks, threads>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
											   numRows, numCols, d_outImage);
	// Synchronize
	cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());
	
	cudaError_t err;
	while ( (err = cudaGetLastError()) != cudaSuccess )
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void cleanup()
{
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_redBlurred));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_greenBlurred));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_blueBlurred));
}

