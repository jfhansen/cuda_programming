/**************************************************************
 * File: im_processing.cpp
 * Description: Implementation of functions that preprocess and 
 * postprocess image files, i.e. load in image files into arrays
 * of uchar4, and save arrays of unsigned chars into image files.
 *
 * Author: jfhansen
 * Last Modification: 29/07/2020
 *************************************************************/

#include <opencv2/opencv.hpp>
#include "im_processing.hpp"

// Declaration OpenCV input image object
cv::Mat inImage;
// Declaration OpenCV output image object
cv::Mat outImage;

// Declaration of pointers to input image arrays in host memory
uchar4 *d_inImage;
// Declaration of pointers to output image arrays in host memory
unsigned char *d_outImage;

size_t numRows() { return inImage.rows; }

size_t numCols() { return inImage.cols; }

void preProcess(const std::string &input_file, uchar4 **d_inputImage, 
    unsigned char **d_outputImage)
{
    // Load in image from input_file
    cv::Mat image;
    image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    // Convert image from BGR format to RGBA
    cv::cvtColor(image, inImage, CV_BGR2RGBA);

    // Create output image
    outImage.create(numRows(), numCols(), CV_8UC1);

    // Allocate memory on GPU for input image and output image
    const size_t numPixels = numRows() * numCols();
    cudaMalloc(d_inputImage, sizeof(uchar4) * numPixels);
    cudaMalloc(d_outputImage, sizeof(unsigned char) * numPixels);
    cudaMemset(*d_outputImage, 0, sizeof(unsigned char) * numPixels);
    // Copy input image to device memory
    cudaMemcpy( *d_inputImage, (uchar4*)inImage.ptr<unsigned char>(0), numPixels*sizeof(uchar4), cudaMemcpyHostToDevice);
    d_inImage = *d_inputImage;
    d_outImage = *d_outputImage;
}

void postProcess(const std::string &output_file)
{
    // Compute number of pixels in image.
    const int numPixels = numRows() * numCols();
    // Copy array from device memory to the OpenCV object, outImage
    cudaMemcpy(outImage.ptr<unsigned char>(0), d_outImage, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // Write outImage to output_file.
    cv::imwrite(output_file.c_str(), outImage);
    // Cleanup device memory
    cudaFree(d_inImage);
    cudaFree(d_outImage);
}
