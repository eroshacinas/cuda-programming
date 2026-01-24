#include <stdio.h>
#include <cuda_runtime.h> // cuda header

#define BLUR_SIZE 1 // 3x3 blur filter

int main() {
    printf("blur image filter\n");

    return 0;
}


__global__ void blurKernel(
    unsigned char *in,
    unsigned char *out,
    int w,
    int h
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < h && col < w) {
        int pixVal = 0;
        int pixels = 0;

        // get average of surrounding BLUR_SIZE x BLUR_SIZE box
        // blurRow = -1, 0, 1
        // blurCol = -1, 0, 1
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // verify we have a valid image pixel
                if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow*w + curCol]; // accumulate neighbor pixel values
                    ++pixels; // keep track of number of pixels in avg
                }
            }
        }

        // write our new pixel value out
        out[row*w + col] = (unsigned char) (pixVal/pixels);
    }
}