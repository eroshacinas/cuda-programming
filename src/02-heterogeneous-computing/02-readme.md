# CUDA Programming

## Chapter 02 - Heterogeneous Computing

### Memory Management API

#### `cudaMalloc()`
Allocates object in the device global memory.
- **Parameters:**
  1. Pointer to the pointer address (e.g., `(void **)&d_A`).
  2. Size of the allocated object in bytes.
- **Note:** The address of the pointer variable should be cast to `(void **)` because the function expects a generic pointer. This allows `cudaMalloc` to write the address of the allocated memory into the provided pointer regardless of its type.
- **Calculation:** `num_bytes = array_size * sizeof(type)`

#### `cudaFree()`
Frees object from device global memory.
- **Parameters:** Pointer to the object to be freed.

#### `cudaMemcpy()`
Copies data between memory spaces.
- **Parameters:**
  1. **Destination:** Pointer to destination location.
  2. **Source:** Pointer to source location.
  3. **Size:** Number of bytes to be copied.
  4. **Direction:** Type of memory transfer (e.g., `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`).

---

### Kernel Definition

#### `__global__`
- This keyword indicates that the function is a **kernel**.
- It is called from the host to generate a grid of threads on a device.

---

### Threads and Blocks Concepts

> **Performance Tip:** Each thread block can have up to 1024 threads (in CUDA 3.0+). It is advised to use multiples of **32 threads** per block for optimal performance.

#### Coordinate Definitions

CUDA kernels have access to special variables `threadIdx` and `blockIdx`:
- They allow threads to distinguish themselves from each other.
- They determine the area of data each thread will work on.

| Variable | Definition | Analogy |
| :--- | :--- | :--- |
| **`threadIdx`** | A unique coordinate within a specific block. | Local Phone Number |
| **`blockIdx`** | A common coordinate for all threads in a block. | Area Code |

#### Calculating Global Indices

Each thread can combine its `threadIdx` and `blockIdx` to create a unique global index within the entire grid.

**Formula:**
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

*   `blockDim.x * blockIdx.x`: Calculates the **base** (start of the block).
*   `threadIdx.x`: Calculates the **offset** (position within the block).

**Example:**
*Assuming a block size (`blockDim`) of 256 threads:*

1.  **Block 0 (`blockIdx = 0`):**
    *   `threadIdx` range: 0 to 255.
    *   Global Index `i`: 0 to 255.
2.  **Block 1 (`blockIdx = 1`):**
    *   `threadIdx` range: 0 to 255.
    *   Global Index `i`: 256 to 511.

This allows mapping massive arrays to parallel threads efficiently.