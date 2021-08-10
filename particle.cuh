#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES

#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "config.h"
#include "random.cuh"
#include "trailmap.cuh"

#ifndef __PARTICLE__
#define __PARTICLE__

class particle
{
public:
    float heading;
    float location[2];
    int speciesMask[3];
    particleConfig* config;
    modelConfig* mdlConfig;

    __device__
    void initialize(int seed, curandState* devState, particleConfig* cfg, modelConfig* mdlCfg);

    __device__
    float sense(float angleOffset, trailMap tm, curandState* devState);

    __device__
    void move(trailMap tm, curandState* devState);

    __device__
    void update(trailMap tm, curandState* devState);
};

#endif