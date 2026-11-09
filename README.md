Colab link: [https://colab.research.google.com/drive/1PIgWy9WmFAw5WUFPmaC9DaQngOWD_xei?usp=sharing](https://colab.research.google.com/drive/1PIgWy9WmFAw5WUFPmaC9DaQngOWD_xei?usp=sharing)
# ASUM: Sum of Absolute Values

ASUM computes the sum of the absolute values of a vector of `n` double-precision floats:

$$
\text{asum} = \sum_{i=0}^{n-1} |A_i|
$$


## Benchmark Setup

| | |
| :--- | :--- |
| Machine | |
| OS |  |
| CPU | |
| Cache | |
| RAM | |
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
