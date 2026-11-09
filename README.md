Colab link: [https://colab.research.google.com/drive/1PIgWy9WmFAw5WUFPmaC9DaQngOWD_xei?usp=sharing](https://colab.research.google.com/drive/1PIgWy9WmFAw5WUFPmaC9DaQngOWD_xei?usp=sharing)
# ASUM: Sum of Absolute Values

ASUM computes the sum of the absolute values of a vector of `n` double-precision floats:

$$
\text{asum} = \sum_{i=0}^{n-1} |A_i|
$$


## Benchmark Setup

| | |
| :--- | :--- |
| Machine | Google Colab (Google Compute Engine VM, KVM) |
| OS | Ubuntu 24.04.5 LTS x86_64, Linux kernel 6.6.159+ |
| CPU | Intel Xeon @ 2.00GHz (Skylake-SP, family 6 model 85), 1 core / 2 threads, AVX2 + AVX-512 |
| GPU | NVIDIA Tesla T4 |
| Cache | L1d 32 KiB, L1i 32 KiB, L2 1 MiB, L3 38.5 MiB |
| RAM | 12.7 GiB (12975 MiB) |
| Vector init | |
| # of runs | 30 

## Program Output

## Analysis

#### Table

| CUDA block size = | Kernel time [avg \| min \| max] | Speedup vs baseline C program |
| :--- | :--- | :--- |
| C |  |  |
| x86-64 |  |  |
| X86 64 SIMD XMM |  |  |
| x86 64 SIMD YMM |  |  |
| CUDA Unified |  |  |
| CUDA Prefetch |  |  |
| CUDA Prefetch+page creation |  |  |
| CUDA Prefetch+Page creation+memadvise |  |  |

### SIMD vs SIMT in handling parallelism

### PROS and CONS of using SIMD and SIMT

### How block size affects execution time and recommendation


### Is prefetching always recommended, or should CUDA manage memory?


### SIMD vs SIMT

## Problems Encountered and AHA Moments

### Problems

### AHA Moments

## Conclusion
