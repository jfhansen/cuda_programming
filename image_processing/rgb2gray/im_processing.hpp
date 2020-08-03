/**************************************************************
 * File: im_processing.hpp
 * Description: Definition of functions that preprocess and 
 * postprocess image files, i.e. load in image files into arrays
 * of uchar4, and save arrays of unsigned chars into image files.
 *
 * Author: jfhansen
 * Last Modification: 29/07/2020
 *************************************************************/

#ifndef IM_PROCESSING_HPP
#define IM_PROCESSING_HPP

#include <opencv2/core/core.hpp>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

size_t numRows();
size_t numCols();

void preProcess(const std::string &input_file, uchar4 **d_inputImage,
    unsigned char **d_outputImage);

void postProcess(const std::string &output_file);

#endif