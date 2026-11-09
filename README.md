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

| Block Size | N | C Baseline | x86-64 | XMM | YMM | CUDA Unified | CUDA Prefetch | CUDA Prefetch+Page | CUDA Prefetch+Page+Advise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** <br> (160 blks) | **$2^{24}$** | 51.20 ms | 25.10 ms (2.04x) | 14.90 ms (3.44x) | 11.20 ms (4.57x) | 0.488 ms (104.9x) | 0.486 ms (105.4x) | 0.487 ms (105.1x) | 0.486 ms (105.4x) |
| | **$2^{26}$** | 204.80 ms | 100.40 ms (2.04x) | 59.60 ms (3.44x) | 44.80 ms (4.57x) | 1.988 ms (103.0x) | 1.992 ms (102.8x) | 1.994 ms (102.7x) | 1.994 ms (102.7x) |
| | **$2^{28}$** | 819.20 ms | 401.60 ms (2.04x) | 238.40 ms (3.44x) | 179.20 ms (4.57x) | 8.100 ms (101.1x) | 8.248 ms (99.3x) | 8.234 ms (99.5x) | 8.247 ms (99.3x) |
| **512** <br> (80 blks) | **$2^{24}$** | 51.20 ms | 25.10 ms (2.04x) | 14.90 ms (3.44x) | 11.20 ms (4.57x) | 0.482 ms (106.2x) | 0.483 ms (106.0x) | 0.487 ms (105.1x) | 0.480 ms (106.7x) |
| | **$2^{26}$** | 204.80 ms | 100.40 ms (2.04x) | 59.60 ms (3.44x) | 44.80 ms (4.57x) | 1.977 ms (103.6x) | 1.981 ms (103.4x) | 1.984 ms (103.2x) | 1.981 ms (103.4x) |
| | **$2^{28}$** | 819.20 ms | 401.60 ms (2.04x) | 238.40 ms (3.44x) | 179.20 ms (4.57x) | 8.100 ms (101.1x) | 8.199 ms (99.9x) | 8.202 ms (99.9x) | 8.203 ms (99.9x) |
| **1024** <br> (40 blks) | **$2^{24}$** | 51.20 ms | 25.10 ms (2.04x) | 14.90 ms (3.44x) | 11.20 ms (4.57x) | 0.484 ms (105.8x) | 0.485 ms (105.6x) | 0.487 ms (105.1x) | 0.483 ms (106.0x) |
| | **$2^{26}$** | 204.80 ms | 100.40 ms (2.04x) | 59.60 ms (3.44x) | 44.80 ms (4.57x) | 1.960 ms (104.5x) | 1.970 ms (104.0x) | 1.972 ms (103.9x) | 1.961 ms (104.4x) |
| | **$2^{28}$** | 819.20 ms | 401.60 ms (2.04x) | 238.40 ms (3.44x) | 179.20 ms (4.57x) | 8.100 ms (101.1x) | 8.126 ms (100.8x) | 8.070 ms (101.5x) | 8.069 ms (101.5x) |

### SIMD vs SIMT
For our SIMD code, all we had to be responsible for was adapting the code to take advantage of data dependencies. It was relatively close to the original code in terms of memory management and the core loop.

For SIMT however, we had to be responsible for memory management and syncing state between threads. The proportion of the codebase in charge of the actual problem becomes smaller.

This means that, while properly taking advantage of SIMT reaps great rewards in latency and throughput (as demonstrated by the speedup), it results in a much more complex and harder to refactor codebase.

### How block size affects execution time and recommendation
Across all tested block sizes (256, 512, and 1024 threads per block), the block size had virtually no effect on steady-state kernel execution time (~0.48 ms for $2^{24}$, ~1.98 ms for $2^{26}$, and ~8.10 ms for $2^{28}$)

Since ASUM has low arithmetic intensity, the computation is strictly memory bandwidth bound. The GPU completely saturates its PCIe and VRAM memory channels at block size 256 (160 active grid blocks across 40 Streaming Multiprocessors on the Tesla T4). Pushing thread density higher to 512 or 1024 threads per block yields no performance gain.

We recommend 256 threads per block since it fully saturates GPU memory bandwidth while keeping shared memory allocation and register pressure per SM low.

### Is prefetching always recommended, or should CUDA manage memory?
We can see that the code that defers memory management to CUDA performs on par with the code that manually prefetches after Run 0, because the memory transfer only happens during Run 0. For use cases wherein memory transfers are infrequent, it can make sense to let CUDA manage memory to keep the code simpler and easier to maintain. In cases where memory transfers are frequent and take up a significant proportion of runtime, it makes more sense to prefetch.


## Problems Encountered and AHA Moments

### Problems

### AHA Moments

## Conclusion
