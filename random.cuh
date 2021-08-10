#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_math.h>

#ifndef __RANDOM__
#define __RANDOM__

__device__
void genRandom(float* a, curandState* devState);

__global__
void initCurand(unsigned long seed, curandState* devState);

#endif