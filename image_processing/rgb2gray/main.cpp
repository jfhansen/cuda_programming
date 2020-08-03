/**************************************************************
 * File: main.cpp
 * Description: Main file
 *
 * Author: jfhansen
 * Last Modification: 29/07/2020
 *************************************************************/

#include <iostream>
#include "im_processing.hpp"

void rgba_to_grayscale(const uchar4 *const d_rgbaImage, unsigned char *const d_grayImage,
                        size_t numRows, size_t numCols);

int main(int argc, char* argv[])
{
    uchar4 *d_rgbaImage;
    unsigned char *d_grayImage;
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
    preProcess(input_file, &d_rgbaImage, &d_grayImage);
    
    // Launch kernel
    rgba_to_grayscale(d_rgbaImage, d_grayImage, numRows(), numCols());

    // Save image
    postProcess(output_file);

    return 0;
}