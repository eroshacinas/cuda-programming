#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> // cuda header

#define WIDTH 3 // matrix width and height

// assume square block and grid 
#define BLOCK_SIZE 4 
#define GRID_SIZE 4

/*
    TODO:
    1. optimize block and grid size to minimize unused threads depending on matrix shape
    2. make matrix irregular
    3. further optimize matmult kernel: use another set of threads to eliminate for loop? its a dot product after all
*/


// func prototypes
void printMatrix(float M[WIDTH][WIDTH]);
float* linearizeMatrix(float M[WIDTH][WIDTH]);
void printLinearMatrix(float *M);
void initializeMatrix(float M[WIDTH][WIDTH]);
void transpose2DMatrix(float in[WIDTH][WIDTH], float out[WIDTH][WIDTH]);

__global__ void MatrixMulKernel(float *M, float *N, float *P,int Width);
void MatrixMul(float *M_h, float *N_h, float *P_h);


int main(){
    printf("matrix multiplication\n");
    // initialize 2D matrix. linearize later
    float M[WIDTH][WIDTH] = {0}, N[WIDTH][WIDTH] = {0}, N_T[WIDTH][WIDTH] = {0}, P[WIDTH][WIDTH] = {0};
    initializeMatrix(M);
    initializeMatrix(N);

    // printMatrix(M);
    // printMatrix(N);

    // tanspose N
    transpose2DMatrix(N, N_T);

    float *linearM = linearizeMatrix(M);
    float *linearN_T = linearizeMatrix(N_T);
    float *linearP = linearizeMatrix(P);

    

    // printMatrix(N_T);

    MatrixMul(linearM, linearN_T, linearP);

    printLinearMatrix(linearP);

    return 0;
}


// accept linearized matrices
void MatrixMul(
    float *M_h,
    float *N_h,
    float *P_h
) {
    float *M_d, *N_d, *P_d;

    int size = WIDTH * WIDTH * sizeof(float);
    
    // allocate mem in gpu
    cudaMalloc((void**) &M_d, size);
    cudaMalloc((void**) &N_d, size);
    cudaMalloc((void**) &P_d, size);

    // transfer to gpu
    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    
    // launch kernel
    dim3 dimGrid(GRID_SIZE, GRID_SIZE, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, WIDTH);

    // copy result to host
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    // free mem
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

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


void printMatrix(float M[WIDTH][WIDTH]) {
    int i, j=0;

    for(i=0; i<WIDTH; i++) {
        for (j = 0; j<WIDTH; j++) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
}


void printLinearMatrix(float *M){
    int i = 0;

    for(i = 0; i < WIDTH*WIDTH; i++){
        printf("%f ", M[i]);

        if((i+1) % WIDTH == 0)
            printf("\n");
    }
}


float* linearizeMatrix(float M[WIDTH][WIDTH]){
    float *linearM = (float *) malloc(WIDTH * WIDTH * sizeof(float));
    int i, j, index=0;

    for(i=0; i<WIDTH; i++) {
        for(j=0; j<WIDTH; j++) {
            linearM[index++] = M[i][j];
        }
    }
    return linearM;
}


void initializeMatrix(float M[WIDTH][WIDTH]){
    int i, j;
    int val = 1;
    for(i=0; i < WIDTH; i++) {
        for(j=0; j < WIDTH; j++, val++){
            M[i][j] = val;
        }
    }

}


void transpose2DMatrix(float in[WIDTH][WIDTH], float out[WIDTH][WIDTH]){
    int i, j;
    for(i=0; i<WIDTH; i++){
        for(j=0; j<WIDTH; j++){
            out[j][i] = in[i][j];
        }
    } 
}