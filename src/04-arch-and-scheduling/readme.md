## Chapter 04 - GPU Architecture and Scheduling

- GPU architecture is organized into an array of highly threaded streaming multiprocessors (SMs)

- Each SM has several processing units called streaming processors or CUDA cores (or just cores)

- Each SM has its own control and on-chip memory (which differs from global memory or VRAM)

![GPU Architecture](images/gpu-arch.png)

### Block Assignment

- Multiple blocks are likely to be simultaneously assigned to the same SM

- However, blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM

- There's a limit on the total number of blocks that can be simulteneously executing in a CUDA device

- The assignment of threads to SMs on a block-by-block basis guarantees that threads in the same block are scheduled simultaneously on the same SM

- This makes it possible for threads in the same block to interact with each other in ways that threads across different blocks can't.

![Block Assignment](images/block-assignment.png)

### Synchronization and Transparent Scalability

- CUDA allows threads in the same block to coordinate activities using barrier sync func `__syncthreads()`

- `__syncthreads()` must be executed by **all** threads in a block

![Barrier Synchronization](images/barrier-sync.png)

- if `__syncthreads()` is placed in an if statement, either all threads in a block execute the path that includes the `__syncthreads()` or **none** of them does

- an incorrect use is shown below

```C
void incorrect_barrier_example(int n) {
    ...
    if(threadIdx.x % 2 == 0){
        ...
        __syncthreads{};
    } else {
        ...
        __syncthreads{};
    }
}
```

- the code above violates the rule of threads executing at the same line where `__syncthreads{}` is called. THis results in an undefined behavior

- In general, incorrect usage of barrier synchronization can result in incorrect result or a deadlock

- The trade-off of not allowing barrier synchronization of threads in different blocks leads to transparent scalability as seen below

- If blocks could synchronize with each other: The runtime would need to schedule all blocks that need to sync together at the same time, which would require enormous resources and limit how the GPU could execute your code.

- By preventing inter-block synchronization: Blocks become completely independent execution units. The runtime can:

* Execute blocks in any order (Block 0 then Block 1, or Block 5 then Block 2, etc.)
* Execute any number of blocks simultaneously based on available resources
* Run the same kernel on a cheap GPU with few cores or an expensive GPU with many cores

- From the figure:
* Left side (low-cost GPU): Only 2 blocks execute simultaneously because this GPU has limited execution resources (SMs, registers, shared memory)
* Right side (high-end GPU): 4 blocks execute simultaneously because this GPU has more resources

![Barrier Scalability](images/barrier-sync-scalability.png)


