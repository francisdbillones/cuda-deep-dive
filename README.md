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
| Vector init | `A[i] = (i % 1000) - 500` |
| # of runs | 30 

## Program Output
Variant 2 Baseline grid-stride CUDA kernel:
![variant21_output](output/var2_1.PNG)
![variant22_output](output/var2_2.PNG)
Variant 3 Baseline grid-stride CUDA kernel:
![variant31_output](output/var3_1.PNG)
![variant32_output](output/var3_2.PNG)
Variant 4 Grid-stride CUDA kernel with prefetching, page-creation:
![variant41_output](output/var4_1.PNG)
![variant42_output](output/var4_2.PNG)
Variant 5 Grid-stride CUDA kernel with prefetching, page-creation, and mem-advise:
![variant51_output](output/var5_1.PNG)
![variant52_output](output/var5_2.PNG)


## Analysis

![Execution Time Log Scale Output](output/graph1.png)
![Steady State GPU Kernel Output](output/graph2.png)

#### Table

| Block Size | N | C Baseline | x86-64 | XMM | YMM | CUDA Unified | CUDA Prefetch | CUDA Prefetch+Page | CUDA Prefetch+Page+Advise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** <br> (160 blks) | **$2^{24}$** | 51.20 ms | 25.10 ms (2.04x) | 14.90 ms (3.44x) | 11.20 ms (4.57x) | 0.481 ms (106.4x) | 0.486 ms (105.3x) | 0.487 ms (105.1x) | 0.485 ms (105.6x) |
| | **$2^{26}$** | 204.80 ms | 100.40 ms (2.04x) | 59.60 ms (3.44x) | 44.80 ms (4.57x) | 1.994 ms (102.7x) | 1.990 ms (102.9x) | 1.992 ms (102.8x) | 1.990 ms (102.9x) |
| | **$2^{28}$** | 819.20 ms | 401.60 ms (2.04x) | 238.40 ms (3.44x) | 179.20 ms (4.57x) | 8.100 ms (101.1x) | 8.267 ms (99.1x) | 8.238 ms (99.4x) | 8.302 ms (98.7x) |
| **512** <br> (80 blks) | **$2^{24}$** | 51.20 ms | 25.10 ms (2.04x) | 14.90 ms (3.44x) | 11.20 ms (4.57x) | 0.483 ms (106.0x) | 0.483 ms (106.0x) | 0.484 ms (105.8x) | 0.480 ms (106.7x) |
| | **$2^{26}$** | 204.80 ms | 100.40 ms (2.04x) | 59.60 ms (3.44x) | 44.80 ms (4.57x) | 1.979 ms (103.5x) | 1.981 ms (103.4x) | 1.986 ms (103.1x) | 1.972 ms (103.9x) |
| | **$2^{28}$** | 819.20 ms | 401.60 ms (2.04x) | 238.40 ms (3.44x) | 179.20 ms (4.57x) | 8.100 ms (101.1x) | 8.234 ms (99.5x) | 8.196 ms (99.9x) | 8.181 ms (100.1x) |
| **1024** <br> (40 blks) | **$2^{24}$** | 51.20 ms | 25.10 ms (2.04x) | 14.90 ms (3.44x) | 11.20 ms (4.57x) | 0.488 ms (104.9x) | 0.486 ms (105.3x) | 0.486 ms (105.3x) | 0.482 ms (106.2x) |
| | **$2^{26}$** | 204.80 ms | 100.40 ms (2.04x) | 59.60 ms (3.44x) | 44.80 ms (4.57x) | 1.960 ms (104.5x) | 1.968 ms (104.1x) | 1.973 ms (103.8x) | 1.960 ms (104.5x) |
| | **$2^{28}$** | 819.20 ms | 401.60 ms (2.04x) | 238.40 ms (3.44x) | 179.20 ms (4.57x) | 8.100 ms (101.1x) | 8.143 ms (100.6x) | 8.128 ms (100.8x) | 8.080 ms (101.4x) |

#### Effective Memory Bandwidth

The effective bandwidth is computed from the steady-state kernel times above as $\text{BW} = \frac{8N \text{ bytes}}{t}$, since each element of `A` is a `double` read exactly once (the writes to `out` are negligible). Percentages are relative to the T4's theoretical peak of 320 GB/s.

| Block Size | N | Bytes Read | CUDA Unified | CUDA Prefetch | CUDA Prefetch+Page | CUDA Prefetch+Page+Advise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | **$2^{24}$** | 134 MB | 279.0 GB/s (87%) | 276.2 GB/s (86%) | 275.6 GB/s (86%) | 276.7 GB/s (86%) |
| | **$2^{26}$** | 537 MB | 269.2 GB/s (84%) | 269.8 GB/s (84%) | 269.5 GB/s (84%) | 269.8 GB/s (84%) |
| | **$2^{28}$** | 2.15 GB | 265.1 GB/s (83%) | 259.8 GB/s (81%) | 260.7 GB/s (81%) | 258.7 GB/s (81%) |
| **512** | **$2^{24}$** | 134 MB | 277.9 GB/s (87%) | 277.9 GB/s (87%) | 277.3 GB/s (87%) | 279.6 GB/s (87%) |
| | **$2^{26}$** | 537 MB | 271.3 GB/s (85%) | 271.0 GB/s (85%) | 270.3 GB/s (84%) | 272.2 GB/s (85%) |
| | **$2^{28}$** | 2.15 GB | 265.1 GB/s (83%) | 260.8 GB/s (82%) | 262.0 GB/s (82%) | 262.5 GB/s (82%) |
| **1024** | **$2^{24}$** | 134 MB | 275.0 GB/s (86%) | 276.2 GB/s (86%) | 276.2 GB/s (86%) | 278.5 GB/s (87%) |
| | **$2^{26}$** | 537 MB | 273.9 GB/s (86%) | 272.8 GB/s (85%) | 272.1 GB/s (85%) | 273.9 GB/s (86%) |
| | **$2^{28}$** | 2.15 GB | 265.1 GB/s (83%) | 263.7 GB/s (82%) | 264.2 GB/s (83%) | 265.8 GB/s (83%) |

Every configuration sustains 81–87% of peak DRAM bandwidth.


### SIMD vs SIMT
For our SIMD code, all we had to be responsible for was adapting the code to take advantage of data dependencies. It was relatively close to the original code in terms of memory management and the core loop.

For SIMT however, we had to be responsible for memory management and syncing state between threads. The proportion of the codebase in charge of the actual problem becomes smaller.

This means that, while properly taking advantage of SIMT reaps great rewards in latency and throughput (as demonstrated by the speedup), it results in a much more complex and harder to refactor codebase.

### How block size affects execution time and recommendation
Across all tested block sizes (256, 512, and 1024 threads per block), the block size had virtually no effect on steady-state kernel execution time (~0.48 ms for $2^{24}$, ~1.98 ms for $2^{26}$, and ~8.10 ms for $2^{28}$). Since ASUM performs relatively simple arithmetic operations compared to the amount of data it processes, its performance is likely limited by memory bandwidth rather than computational throughput. Based on the results, increasing the block size from 256 to 512 or 1024 threads does not provide a noticeable performance improvement, suggesting that additional threads are not helping to increase memory throughput.

We recommend 256 threads per block as a practical configuration since it achieves nearly identical execution times to the larger block sizes while allowing more blocks to be resident across the GPU's 40 streaming multiprocessors. Since larger block sizes offer no consistent performance advantage, 256 threads provides a reasonable balance between resource utilization and execution time. 


### Is prefetching always recommended, or should CUDA manage memory?
We can see that the code that defers memory management to CUDA performs on par with the code that manually prefetches after Run 0, because the memory transfer only happens during Run 0. For use cases wherein memory transfers are infrequent, it can make sense to let CUDA manage memory to keep the code simpler and easier to maintain. In cases where memory transfers are frequent and take up a significant proportion of runtime, it makes more sense to prefetch.


## Problems Encountered and AHA Moments

### Problems

We took parallel reduction kernel from this [https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf](slide deck from Nvidia), but it wasn't adapted for `blockSize >= 1024`. We added an additional step to the code to account for the possibility of `blockSize >= 1024`.


### AHA Moments

We noticed that for Run >= 1, all the variants performed basically the same. This is proven by our graphs above that show that their runtimes look like random noise. The only noticeable change is in Run 0 when we add prefetching. Without prefetching, the page fault migration time is included in the kernel time, which is why the baseline is so much slower than every other variant. However, once the data is resident on the GPU, all variants converge to the same performance.

Building on top of that, **`PREFETCH_OUT` and `MEM_ADVISE` add nothing measurable here.** `out` is only a few KB, and `asum` is just one read. Memory advice pays off for reuse or contention patterns, but for this project we don't have either of those.

**Block size barely matters once the grid is sized for occupancy.** Going from 256 to 1024 threads per block didn't make the kernel meaningfully faster (~0.48 ms for $N = 2^{24}$). The kernel is bandwidth-bound and the reduction tree only runs once per block, so the extra threads have nothing to speed up. This goes against the intuition we took from the aforementioned slide deck from Nvidia.

## Conclusion
