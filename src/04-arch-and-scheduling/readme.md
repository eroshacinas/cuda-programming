# 🚀 Chapter 04 - GPU Architecture and Scheduling

---

## 📐 GPU Architecture Overview

**GPU architecture is organized into an array of highly threaded Streaming Multiprocessors (SMs)**

Each **SM (Streaming Multiprocessor)** contains:
- 🔹 **Streaming Processors** / **CUDA Cores** (or simply "cores")
- 🔹 **Control Unit**
- 🔹 **On-chip Memory** (distinct from global memory/VRAM)

![GPU Architecture](images/gpu-arch.png)

---

## 🧩 Block Assignment

### Key Concepts

- ✅ **Multiple blocks** are likely to be simultaneously assigned to the same SM
- ⚠️ Blocks require hardware resources, so only a **limited number** can be simultaneously assigned to any given SM
- 📊 There's a limit on the total number of blocks that can simultaneously execute on a CUDA device

### Assignment Guarantees

Assignment of threads to SMs occurs on a **block-by-block basis**, guaranteeing:

1. All threads in the same block are **scheduled simultaneously**
2. All threads in the same block **execute on the same SM**

> 💡 This enables threads within a block to interact in ways that threads across different blocks cannot

![Block Assignment](images/block-assignment.png)

---

## 🔄 Synchronization and Transparent Scalability

### Barrier Synchronization

CUDA allows threads in the same block to coordinate activities using:

```c
__syncthreads()
```

> ⚠️ **Critical Rule**: `__syncthreads()` must be executed by **ALL** threads in a block

![Barrier Synchronization](images/barrier-sync.png)

### ✅ Correct Usage Rules

If `__syncthreads()` is placed within an `if` statement:
- Either **ALL** threads in a block execute the path that includes `__syncthreads()`
- OR **NONE** of them do

### ❌ Incorrect Usage Example

```c
void incorrect_barrier_example(int n) {
    // ...existing code...
    if (threadIdx.x % 2 == 0) {
        // ...existing code...
        __syncthreads();  // ❌ WRONG!
    } else {
        // ...existing code...
        __syncthreads();  // ❌ WRONG!
    }
}
```

> ⚠️ **Why this is wrong**: The code violates the rule that all threads must execute `__syncthreads()` at the same program point. This results in **undefined behavior**.

**Consequences of incorrect usage:**
- 💥 Incorrect results
- 🔒 Deadlock

---

### 🎯 Transparent Scalability

The trade-off of **not allowing** barrier synchronization between different blocks enables **transparent scalability**.

#### ❌ If blocks could synchronize with each other:
- Runtime would need to schedule all blocks requiring synchronization at the same time
- Would require enormous resources
- Would limit how the GPU could execute your code

#### ✅ By preventing inter-block synchronization:

Blocks become completely **independent execution units**, allowing the runtime to:

1. 🔀 Execute blocks in **any order** (e.g., Block 0 → Block 1, or Block 5 → Block 2)
2. ⚡ Execute **any number** of blocks simultaneously based on available resources
3. 🌐 Run the same kernel on both **low-end and high-end GPUs**

**Visual Example:**

| Low-Cost GPU | High-End GPU |
|-------------|--------------|
| 2 blocks execute simultaneously | 4 blocks execute simultaneously |
| Limited execution resources | Greater available resources |

![Barrier Scalability](images/barrier-sync-scalability.png)

> 💡 **Transparent scalability** = Execute the same application program on different hardware with **zero code changes**

---

## 🌊 Warps and SIMD Hardware

### Understanding Warps

> 📌 Conceptually, one should assume that threads in a block can execute in **any order** with respect to each other.

#### Key Definitions

- 📦 **Warp**: A 32-thread unit for scheduling in SMs
- 🔢 **Warp Size**: Fixed at **32 threads** (implementation-specific, may vary in future GPUs)
- 📋 **Thread Organization**: Consecutive `threadIdx` values (0-31 → first warp, 32-63 → second warp, etc.)

### Warp Calculation Examples

```
Block with 256 threads:
  → 256/32 = 8 warps per block
  → With 3 blocks in SM: 8 × 3 = 24 warps total

Block with 48 threads:
  → 2 warps (second warp padded with 16 inactive threads)
```

> ⚠️ For blocks whose size is **not a multiple of 32**, the last warp will be padded with inactive threads to fill up the 32 thread positions.

![Warp-Partitioned Blocks](images/warps.png)

### Multi-Dimensional Thread Blocks

For blocks with multiple dimensions of threads:
- Dimensions are projected into a **linearized row-major layout**
- Then partitioned into warps

![Warp-Partitioned Blocks](images/2d-warp-linearized.png)

![SM for SIMD Execution](images/sm-architecture.png)

---

## 🔀 Control Divergence

### What is Control Divergence?

Threads in the same warp that follow **different execution paths** exhibit **control divergence**.

### How It Works

For an `if-else` construct:
1. If some threads follow the `if`-path and others follow the `else`-path
2. Hardware takes **two passes**:
   - 🔹 Pass 1: Execute threads following the `if`-path
   - 🔹 Pass 2: Execute threads following the `else`-path
3. During each pass, threads following the other path are **inactive**

![Threads Diverging on If-Else](images/threads-conditional-diverge.png)

---

## ⚡ Warp Scheduling and Latency Tolerance

### The Challenge

- SMs have **limited execution units** to execute only a subset of assigned threads at any point in time
- Recent designs: Each SM can execute instructions for a **small number of warps** at any given point in time

### The Solution: Latency Hiding

> 💡 Assigning many more warps to an SM than it can execute at once is how GPUs tolerate **long-latency operations** (e.g., global memory accesses).

**Analogy**: The SM schedules many more warps than it can execute at once so that when one warp hits a **'red light'** (waiting for data from VRAM), it can switch to a **'ready' warp** in **zero clock cycles**, ensuring the math units never sit idle.

### 🔧 Zero-Overhead Context Switching

The GPU achieves **zero-cycle switching** through a "brute force" hardware strategy:

| 💻 CPU | 🎮 GPU |
|--------|--------|
| One set of registers | Massive Register File |
| Must save/restore state to RAM | All warp states stored on-chip simultaneously |
| Context switch overhead | Zero-overhead switching |

**How it works:**
- When a Block is assigned to an SM, the hardware **carves out a permanent slice of registers** for those threads
- They stay there until the Block is **completely finished**

### 📊 Key Constraints

```
Block Limits:
  ✓ Max 1,024 threads per block (regardless of x, y, z dimensions)
  ✓ Fixed warp size of 32 threads

Register File:
  ✓ 65,536 (64K) 32-bit registers per SM
  ✓ SM throws error if it can't fit all states
```

> 🔬 **Why not increase thread limits?** Doubling threads per block increases hardware complexity:
> - `__syncthreads()` for 1,024 threads is manageable
> - For 10,000 threads would require massive "wait logic" circuitry
> - That space is better used for math units

**Latency Tolerance Principle:**
> 🎯 For effective latency tolerance, an SM should have **many more threads assigned** than can be simultaneously executed, maximizing the chance of finding a ready warp at any point in time.

---

## 📊 Resource Partitioning and Occupancy

### Occupancy Definition

```
Occupancy = (Number of warps assigned to SM) / (Maximum warps SM supports)
```

### 🛠️ Execution Resources

An SM's resources are dynamically partitioned across threads:

1. 📝 **Registers**
2. 💾 **Shared Memory**
3. 🎫 **Thread Block Slots**
4. 👥 **Thread Slots**

### Example: Ampere A100 GPU

```
Hardware Limits:
  • Max 32 blocks per SM
  • Max 64 warps (2,048 threads) per SM
  • Max 1,024 threads per block
  • 65,536 (64K) 32-bit registers per SM
```

> ⚠️ **Both max blocks and warps per SM are independent hardware limits. Whichever one you hit first is your "bottleneck."**

### 🍽️ Restaurant Analogy

Imagine a restaurant (the SM) with:
- 🪑 **Total Seats**: 2,048 (Max Threads)
- 🏷️ **Total Tables**: 32 (Max Blocks)

#### Scenario A: Huge Groups (1,024 threads per block)

```
Group 1: 1,024 seats used, 1 table used
Group 2: 2,048 seats used, 2 tables used
Result: ❌ Out of seats! (30 empty tables left)
```

#### Scenario B: Tiny Groups (32 threads per block)

```
32 groups × 32 people = 1,024 people
Result: ❌ All 32 tables used, but 1,024 seats empty!
       (SM only has 32 "check-in" slots for block metadata)
```

### Calculating Full Occupancy

```
For full occupancy:
  65,536 registers / 2,048 threads = 32 registers per thread
```

> ✅ Each thread should use **no more than 32 registers** for full occupancy

---

## 🔍 Querying Device Properties

### Getting Device Count

```c
int devCount;
cudaGetDeviceCount(&devCount);
```

### Iterating Through Devices

```c
cudaDeviceProp devProp;
for(unsigned int i = 0; i < devCount; i++) {
    cudaGetDeviceProperties(&devProp, i);
    // Decide if device has sufficient resources/capabilities
}
```

### 📋 Important Properties

The `cudaDeviceProp` struct contains fields for device properties:

| Property | Description |
|----------|-------------|
| `devProp.maxThreadsPerBlock` | Maximum threads per block |
| `devProp.multiProcessorCount` | Number of SMs in the device |
| `devProp.clockRate` | Clock frequency of the device |
| `devProp.maxThreadsDim[0]` | Max threads in x dimension |
| `devProp.maxThreadsDim[1]` | Max threads in y dimension |
| `devProp.maxThreadsDim[2]` | Max threads in z dimension |

---