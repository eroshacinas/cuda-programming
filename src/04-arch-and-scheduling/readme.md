## Chapter 04 - GPU Architecture and Scheduling

### GPU Architecture Overview

- GPU architecture is organized into an array of highly threaded **Streaming Multiprocessors (SMs)**

- Each SM contains several processing units called:
  - Streaming Processors
  - CUDA Cores (or simply "cores")

- Each SM has its own:
  - Control unit
  - On-chip memory (distinct from global memory/VRAM)

![GPU Architecture](images/gpu-arch.png)

---

### Block Assignment

- **Multiple blocks** are likely to be simultaneously assigned to the same SM

- Blocks require hardware resources to execute, so only a **limited number** of blocks can be simultaneously assigned to any given SM

- There's a limit on the total number of blocks that can simultaneously execute on a CUDA device

- Assignment of threads to SMs occurs on a **block-by-block basis**, guaranteeing that:
  - All threads in the same block are scheduled simultaneously
  - All threads in the same block execute on the same SM
  
- This enables threads within a block to interact in ways that threads across different blocks cannot

![Block Assignment](images/block-assignment.png)

---

### Synchronization and Transparent Scalability

#### Barrier Synchronization

- CUDA allows threads in the same block to coordinate activities using the barrier synchronization function: **`__syncthreads()`**

- **Important**: `__syncthreads()` must be executed by **all** threads in a block

![Barrier Synchronization](images/barrier-sync.png)

#### Correct Usage Rules

- If `__syncthreads()` is placed within an `if` statement, either:
  - **All** threads in a block execute the path that includes `__syncthreads()`, OR
  - **None** of them do

#### Incorrect Usage Example

```c
void incorrect_barrier_example(int n) {
    // ...existing code...
    if (threadIdx.x % 2 == 0) {
        // ...existing code...
        __syncthreads();
    } else {
        // ...existing code...
        __syncthreads();
    }
}
```

⚠️ **Why this is wrong**: The code above violates the rule that all threads must execute `__syncthreads()` at the same program point. This results in **undefined behavior**.

- In general, incorrect usage of barrier synchronization can result in:
  - Incorrect results
  - Deadlock

#### Transparent Scalability

The trade-off of **not allowing** barrier synchronization between different blocks enables **transparent scalability**:

**If blocks could synchronize with each other:**
- The runtime would need to schedule all blocks that require synchronization at the same time
- This would require enormous resources
- It would limit how the GPU could execute your code

**By preventing inter-block synchronization:**
- Blocks become completely independent execution units
- The runtime can:
  - Execute blocks in any order (e.g., Block 0 → Block 1, or Block 5 → Block 2)
  - Execute any number of blocks simultaneously based on available resources
  - Run the same kernel on both low-end and high-end GPUs

**Example from the figure:**
- **Left side (low-cost GPU)**: Only 2 blocks execute simultaneously due to limited execution resources (SMs, registers, shared memory)
- **Right side (high-end GPU)**: 4 blocks execute simultaneously thanks to greater available resources

![Barrier Scalability](images/barrier-sync-scalability.png)

---

- transparent scalability is a characteristic of being able to execute the same application program on different hardware, but with zero changes in the code

#### Wards and SIMD hardware

- Conceptually, one should assume that threads in a block can execute in any order with respect to each other.

- Once a block has been assigned to an SM, it is further divided into 32-thread units called `warps`

- The size of warps is implementation specific and can vary in future generations of GPUs.

- A `warp` is the unit of thread scheduling in SMs

