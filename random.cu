#include "random.cuh"

__device__
void genRandom(float* a, curandState* devState) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[0] = curand_uniform(&devState[idx]);
}

// https://stackoverflow.com/questions/22425283/how-could-we-generate-random-numbers-in-cuda-c-with-different-seed-on-each-run
__global__
void initCurand(unsigned long seed, curandState* devState) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &devState[idx]);
}