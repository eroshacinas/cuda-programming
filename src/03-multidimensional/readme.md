## Chapter 03 - Multidimensional grids and data
- All threads in a grid execute the same kernel function. They rely on coordinates (thread indices) to distinguish themselves from each other 
- Two-level Hierarchy: a.) a grid consisting of one or more blocks, accessed via `blockIdx`; b.) a block made up of threads which are accessed via `threadIdx`
- The execution parameters in a kernel call statement specify the dimensions of the grid and dimensions of each block via `gridDim` and `blockDim`. This variable can be arbitrarily named in the HOST code. 
- They are built-in variables in the KERNEL function however.
- In general, a grid is a three-dimensional (3D) array of blocks, and each block is a 3D array of threads
- E.g. `func_name<<<dimGrid, dimBLock>>>(...);`

```C
dim3 dimGrid(16, 16, 1); // grid has 16x16x1 shape or 16*16 blocks
dim3 dimBlock(32, 32, 1); // each block has 32x32 threads or 32*32 threads
vecAddKernel <<<dimGrid, dimBlock>>>(...); // total threads: dimGrid * dimBlock = 16*16*32*32
```

- CUDA C's allowable range of gridDim.x is from 1 to 2^31 -1 and 1 to 2^16-1 for gridDim.y and gridDim.z
- All threads in a block share the same blockIdx.x, blockIdx.y, and blockIdx.z values
- A block's indeces range is 0 to gridDim.x-1, 0 to gridDim.y-1, 0 to gridDim.z-1

- The total size of a block is limited to 1024 threads. These threads can be distributed across three dimensions in any ways as long as the total number of threads does not exceed 1024.

- A grid and its blocks do not need to have the same dimensionality

![Multi-Dimensional Grid](images/multidimensional-grid.png)

- Note that the ordering of the block and thread labels is such that highest dimension comes first.

- vertical row coordinate = blockIdx.y*blockDim.y + threadIdx.y
- horizontal coordinate = blockIdx.x * blockDim.x + threadIdx.x
- plane / z / depth coordinate = blockIdx.z * blockDim.z + threadIdx.z
