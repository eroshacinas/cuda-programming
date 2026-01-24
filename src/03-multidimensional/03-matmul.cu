#include <stdio.h>
#include <cuda_runtime.h> // cuda header


int main(){
    printf("matrix multiplication\n");

    return 0;
}


__global__
void MatrixMulKernel(
    float *M,
    float *N,
    float *P,
    int Width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x & blockDim.x + threadIdx.x;

    // we're assuming square matrices for now
    if ((row < Width) && (col < Width)){
        float Pvalue = 0;
        for(int k = 0; k < Width; ++k){
            // M traverse rows, N traverse columns
            Pvalue += M[row * Width+k] * N[k*Width+col];
        }

        P[row*Width+col] = Pvalue;
    }
}