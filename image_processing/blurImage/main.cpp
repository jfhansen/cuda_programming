/**************************************************************
 * File: main.cpp
 * Description: Main file
 *
 * Author: jfhansen
 * Last Modification: 02/08/2020
 *************************************************************/

#include <iostream>
#include "im_processing.hpp"

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth);

void gaussian_blur(const uchar4 *const d_inImage, uchar4 *const d_outImage,
    size_t numRows, size_t numCols, const size_t filterWidth);

void cleanup();

int main(int argc, char* argv[])
{
    uchar4 *d_inRgbaImage, *d_outRgbaImage;

    float *h_filter;

    int filterWidth;

    std::string input_file, output_file;
    if (argc == 3)
    {
        input_file = std::string(argv[1]);
        output_file = std::string(argv[2]);
    }
    else 
    {
        std::cout << "Usage: <path-to-executable> input_file output_file" << std::endl;
        exit(1);
    }
    // Load image
    preProcess(input_file, &d_inRgbaImage, &d_outRgbaImage, &h_filter, &filterWidth);

    std::cout << "Preprocessing done." << std::endl;

    // Allocate memory and copy to GPU
    allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

    std::cout << "Allocation of memory and copying to GPU done." << std::endl;
    
    // Run main code
    gaussian_blur(d_inRgbaImage, d_outRgbaImage, numRows(), numCols(), filterWidth);

    std::cout << "Gaussian blur done." << std::endl;

    // Save image
    postProcess(output_file);

    std::cout << "Post-processing done." << std::endl;

    return 0;
}