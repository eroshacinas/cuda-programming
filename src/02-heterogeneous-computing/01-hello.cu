#include <stdio.h>
#include <cuda_runtime.h> // cuda header

#define SIZE 1000
#define THREADS_PER_BLOCK 64

// func prototypes
void populateVector(float *v);
void printVector(float *v);
void naiveVecFloatAdd(float *A, float *B, float *C);
void clearVector(float *v);

void cudaVecFloatAdd(float *A_h, float *B_h, float *C_h);
__global__ void cudaVecFloatAddKernel(float *A, float *B, float *C, int n);

// NOTE: each thread block can have up to 1024 threads in CUDA 3.0 and later
// its advised to use multiples of 32 threads per block for optimal performance

int main() {
    printf("hello cuda programming\n");
    

    float A_h[SIZE], B_h[SIZE]; 
    float C_h[SIZE] = {0}; // this initializes all elements to 0

    populateVector(A_h);
    populateVector(B_h);

    // naive vec add
    naiveVecFloatAdd(A_h, B_h, C_h);
    printVector(C_h);
    clearVector(C_h);

    // cuda 
    cudaVecFloatAdd(A_h, B_h, C_h);

    printVector(C_h);



    return 0;
}

void populateVector(float *v){
    int i = 0;
    for(i=0; i<SIZE; i++)
        v[i] = (float)i;
}

void printVector(float *v){
    int i =0;
    for(i=0; i<SIZE;i++)
        printf("%f ", v[i]);

    printf("\n");
}

// reset vector to zero
void clearVector(float *v){
    int i = 0;
    for(i=0; i<SIZE; i++)
        v[i] = 0;
}

void naiveVecFloatAdd(float *A, float *B, float *C){
    int i = 0;

    for(i=0; i<SIZE; i++)
        C[i] = A[i] + B[i];
}


// kernel caller
void cudaVecFloatAdd(float *A_h, float *B_h, float *C_h) {
    // part 1: allocate device memory and transfer data from host to device
    float *A_d, *B_d, *C_d;
    int size_d = SIZE * sizeof(float);

    cudaError_t err = cudaMalloc((void **) &A_d, size_d);


    // error checking like out of mem
    if(err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
        __FILE__, __LINE__);
        
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void **) &B_d, size_d);
    cudaMalloc((void **) &C_d, size_d);

    cudaMemcpy(A_d, A_h, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_d, cudaMemcpyHostToDevice);

    // part 2: launch kernel to do operation on device
    // launch ceil(n/256) blocks of 256 threads each
    // configuration params are in <<<numBlocks, numThreadsPerBlock>>>
    cudaVecFloatAddKernel<<<ceil(SIZE/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(A_d, B_d, C_d, SIZE);


    // part 3: transfer result from device to host and free device memory
    cudaMemcpy(C_h, C_d, size_d, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
}


// CUDA kernels have access to special vars threadIdx and blockIdx
// these allows threads to distinguish themselves from each other
// and to determine area of data each thread will work on

// threadIdx gives each thread a unique coordinate within a block
// blockIdx gives all threads in a block a common block coordinate

// a good analogy: threadIdx is local phone number; blockIdx is area code
// each thread can combine its threadIdx and blockIdx to create a unique global index for itself within the entire grid

// EXAMPLE: a block has 256 threads, blockDim = 256
// first block has range of threadIdx from 0 to 255
// second block has same threadIdx range, but blockIdx is 1
// to calculate a unique global index for each thread:
// idx = blockIdx.x * blockDim.x + threadIdx.x

// this way, threads from first block will have idx from 0 to 255
// threads from second block will have idx from 256 to 511, and so on

__global__
void cudaVecFloatAddKernel(float *A, float *B, float *C, int n){
    // blockDim.x*blockIdx.x calcualtes the base
    // threadIdx.x is the offset
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // this var i is private to each thread
    // if there's 10_000 threads, there will be 10_000 copies of i, one for each thread

    // boundary check: make sure we don't go out of bounds
    // threads with i >= n will do nothing
    // if we dont put this check, threads with i >= n will access invalid memory locations
    if(i < n){
        C[i] = A[i] + B[i];
    }
}


