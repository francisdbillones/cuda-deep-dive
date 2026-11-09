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

| $2^{24}$ CUDA block size = 256 | Kernel time [avg \| min \| max] | Speedup vs baseline C program |
| :--- | :--- | :--- |
| C | 51.20 ms \| 48.10 ms \| 85.30 ms | 1.00x |
| x86-64 |25.10 ms \| 24.80 ms \| 26.20 ms | 2.04x |
| X86 64 SIMD XMM | 14.90 ms \| 14.50 ms \| 15.20 ms | 3.44x|
| x86 64 SIMD YMM | 11.20 ms \| 10.90 ms \| 11.60 ms | 4.57x|
| CUDA Unified |14.65 ms \| 0.488 ms \| 739.94 ms | 3.49x (Avg) / 104.92x (Min) |
| CUDA Prefetch | 3.56 ms \| 0.486 ms \| 8.25 ms | 14.38x (Avg) / 105.35x (Min) |
| CUDA Prefetch+page creation | 3.56 ms \| 0.486 ms \| 8.25 ms | 14.38x (Avg) / 105.35x (Min) |
| CUDA Prefetch+Page creation+memadvise | 3.56 ms \| 0.486 ms \| 8.25 ms | 14.38x (Avg) / 105.35x (Min)|

### SIMD vs SIMT
For our SIMD code, all we had to be responsible for was adapting the code to take advantage of data dependencies. It was relatively close to the original code in terms of memory management and the core loop.

For SIMT however, we had to be responsible for memory management and syncing state between threads. The proportion of the codebase in charge of the actual problem becomes smaller.

This means that, while properly taking advantage of SIMT reaps great rewards in latency and throughput (as demonstrated by the speedup), it results in a much more complex and harder to refactor codebase.

### How block size affects execution time and recommendation


### Is prefetching always recommended, or should CUDA manage memory?


## Problems Encountered and AHA Moments

### Problems

### AHA Moments

## Conclusion
