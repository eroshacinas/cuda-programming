#include <stdio.h>
#include <cuda_runtime.h> // cuda header

#define CHANNELS 3 // RGB 

int main() {
    printf("grayscale image conversion\n");

    return 0;
}

// input image is encoded as unsigned chars [0, 255]
// each pixel is consecutive chars for 3 channels: R, G, B
// L = 0.21*R + 0.72*G + 0.07*B

// NOTE: the linearized image (*Pin) are in an interleaved NOT planar format
// Interleaved: RGBRGBRGB
// Planar: RRR...RGGG...GBBB...B

// example: 2x2 rgb image of WxHxC
// Interleaved: [R00, G00, B00, R01, G01, B01, R10, G10, B10, R11, G11, B11]
// Planar: [R00, R01, R10, R11] [G00, G01, G10, G11] [B00, B01, B10, B11]

// algo:
// select one pixel from gray scale
// select the corresponding rgb pixel from input image. that is 3 consecutive values
__global__
void colortoGrayscaleConvertion(
    unsigned char* Pout, // 1D grayscale output image
    unsigned char* Pin, // 3D image linearized to 1D
    int width, int height) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        
        // process if within image bounds
        if (col < width && row < height) {
            // get 1d offset for grayscale image
            int grayOffset = row * width + col;

            // linear index for rgb pixel
            int rgbOffset = grayOffset * CHANNELS;

            unsigned char r = Pin[rgbOffset]; // red channel
            unsigned char g = Pin[rgbOffset + 1]; // green channel
            unsigned char b = Pin[rgbOffset + 2]; // blue channel

            // rescale and store
            Pout[grayOffset] = 0.21f * r + 0.72f * g + 0.07f * b;
        }
    }