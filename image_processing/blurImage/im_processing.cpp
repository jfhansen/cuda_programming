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
#include <helper_cuda.h>
#include <helper_functions.h>
#include "im_processing.hpp"

// Declaration OpenCV input image object
cv::Mat inImage;
// Declaration OpenCV output image object
cv::Mat outImage;

// Declaration of pointers to input and output image arrays in host memory
uchar4 *d_inImage, *d_outImage;

// Declaration of filter in host memory
float *h_filter__;

size_t numRows() { return inImage.rows; }

size_t numCols() { return inImage.cols; }

void preProcess(const std::string &input_file, uchar4 **d_inputImage, 
    uchar4 **d_outputImage, float **h_filter, int *filterWidth)
{
    // Load in image from input_file
    cv::Mat image;
    image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    // Convert image from BGR format to RGBA
    cv::cvtColor(image, inImage, CV_BGR2RGBA);

    // Create output image
    outImage.create(numRows(), numCols(), CV_8UC4);

    // Allocate memory on GPU for input image and output image
    const size_t numPixels = numRows() * numCols();
    cudaMalloc(d_inputImage, sizeof(uchar4) * numPixels);
    cudaMalloc(d_outputImage, sizeof(uchar4) * numPixels);
    cudaMemset(*d_outputImage, 0, sizeof(uchar4) * numPixels);
    // Copy input image to device memory
    cudaMemcpy(*d_inputImage, (uchar4*)inImage.ptr<unsigned char>(0), 
                    numPixels*sizeof(uchar4), cudaMemcpyHostToDevice);
    d_inImage = *d_inputImage;
    d_outImage = *d_outputImage;

    // Create filter
    const int blurKernelWidth = 9;
    const float blurKernelSigma = 2.f;

    *filterWidth = blurKernelWidth;
    *h_filter = new float[blurKernelWidth*blurKernelWidth];
    h_filter__ = *h_filter;

    float filterSum = 0.0f;

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; r++) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; c++) {
            float filterVal = expf( - (float)(c*c + r*r) / (2.f * blurKernelSigma * blurKernelSigma));
            (*h_filter)[(r+blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterVal;
            filterSum += filterVal;
        }
    }

    float normFactor = 1.f / filterSum;

    for (int r = 0; r < blurKernelWidth; r++)
        for (int c = 0; c < blurKernelWidth; c++)
            (*h_filter)[r*blurKernelWidth + c] *= normFactor;
}

void postProcess(const std::string &output_file)
{
    // Compute number of pixels in image.
    const int numPixels = numRows() * numCols();
    // Copy array from device memory to the OpenCV object, outImage
    cudaMemcpy(outImage.ptr<uchar4>(0), d_outImage, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost);

    cv::Mat outImageBGR;
    cv::cvtColor(outImage, outImageBGR, CV_RGBA2BGR);
    // Write outImage to output_file.
    cv::imwrite(output_file.c_str(), outImageBGR);
    // Cleanup device memory
    cudaFree(d_inImage);
    cudaFree(d_outImage);
    // Cleanup host memory
    delete [] h_filter__;
}