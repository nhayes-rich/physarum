#include <stdlib.h>
#include <stdio.h>

#include "config.h"
#define _USE_MATH_DEFINES

#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <helper_cuda.h>
#include <helper_math.h>

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

// Configs
__device__ particleConfig* globCfg;
__device__ modelConfig* globMdlCfg;
int blockSize = 1024;
int particleNumBlocks;
int trailMapNumBlocks;

// 3d to 1d index
__device__
float get(float* trailMap, int i, int j, int k, modelConfig *mdlCfg) {
    int index = i * 3 + 
                j * mdlCfg->width * 3 +
                k;
    return trailMap[index];
}

__device__
void set(float val, float* trailMap, int i, int j, int k, modelConfig* mdlCfg) {
    int index = i * 3 + 
                j * mdlCfg->width * 3 +
                k;
    trailMap[index] = val;
}

// CUDA random numbers
__device__ curandState* devState;

__device__
void genRandom(float* a, curandState *devState) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[0] = curand_uniform(&devState[idx]);
}

// https://stackoverflow.com/questions/22425283/how-could-we-generate-random-numbers-in-cuda-c-with-different-seed-on-each-run
__global__
void initCurand(unsigned long seed, curandState *devState) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &devState[idx]);
}

// Particles

class particle
{
public:
    float heading;
    float location[2];
    int speciesMask[3];
    particleConfig* config;
    modelConfig* mdlConfig;

    __device__
    void initialize(int seed, curandState *devState, particleConfig *cfg, modelConfig* mdlCfg) {
        config = &(cfg[seed % mdlCfg->numSpecies]);
        mdlConfig = mdlCfg;
        float random;
        genRandom(&random, devState);

        heading = random * 2 * M_PI;

        if (mdlConfig->start == CENTER) {
            location[0] = mdlConfig->width / 2;
            location[1] = mdlConfig->height / 2;
        }
        else if (mdlConfig->start == CIRCLE) {
            float randomCircle;
            genRandom(&randomCircle, devState);
            float rx = mdlConfig->width / 3;
            float ry = mdlConfig->height / 3;

            location[0] = cos(randomCircle * 2 * M_PI) * rx + mdlConfig->width / 2;
            location[1] = sin(randomCircle * 2 * M_PI) * ry + mdlConfig->height / 2;

            heading = randomCircle * 2 * M_PI + M_PI_2;
        }

        if (mdlConfig->numSpecies == 1) {
            speciesMask[0] = 1;
            speciesMask[1] = 1;
            speciesMask[2] = 1;
        }
        else {
            speciesMask[0] = config->index == 0;
            speciesMask[1] = config->index == 1;
            speciesMask[2] = config->index == 2;
        }
       
    }

    __device__
    float sense(float angleOffset, float *trailMap, curandState *devState) {
        float random;
        genRandom(&random, devState);

        float angle = heading + angleOffset;
        float dir[2] = { config->SO * cos(angle), config->SO * sin(angle) };
        float center[2];

        for (int i = 0; i < 2; i++) {
            center[i] = location[i] + dir[i];
        }

        float mask[3];

        for (int k = 0; k < 3; k++) {
            mask[k] = 2 * speciesMask[k] - 1;
        }

        float sum = 0;
        for (int offsetX = -config->SW; offsetX <= config->SW; offsetX++) {
            for (int offsetY = -config->SW; offsetY <= config->SW; offsetY++) {
                int pos[2];
                pos[0] = center[0] + offsetX, pos[1] = center[1] + offsetY;

                if (pos[0] >= 0 && pos[0] < mdlConfig->width && pos[1] >= 0 && pos[1] < mdlConfig->height) {

                    for (int k = 0; k < 3; k++) {
                        sum += get(trailMap, pos[0], pos[1], k, mdlConfig) * mask[k];
                    }
                }
            }
        }
        return sum;
    }

    __device__
    void move(float *trailMap, curandState *devState) {
        float cosHeading = cos(heading);
        float sinHeading = sin(heading);

        float newX = config->SS * cosHeading + location[0];
        float newY = config->SS * sinHeading + location[1];
        int indX, indY;

        if (newX < 0 || newX >= mdlConfig->width || newY < 0 || newY >= mdlConfig->height) {
            float random;
            genRandom(&random, devState);
            heading = random * 2 * M_PI;
        
            newX = MIN(mdlConfig->width - 1, MAX(0, newX));
            newY = MIN(mdlConfig->height - 1, MAX(0, newY));
        }
        
        indX = (int)floor(newX);
        indY = (int)floor(newY);

        location[0] = newX, location[1] = newY;

        if (config->index == -1) {
            set(config->depositT, trailMap, indX, indY, 0, mdlConfig);
            set(config->depositT, trailMap, indX, indY, 1, mdlConfig);
            set(config->depositT, trailMap, indX, indY, 2, mdlConfig);
        }
        else {
            set(config->depositT, trailMap, indX, indY, config->index, mdlConfig);
        }


    }

    __device__
    void update(float* trailMap, curandState *devState) {
        float f = sense(0, trailMap, devState);
        float fl = sense(config->SA, trailMap, devState);
        float fr = sense(-config->SA, trailMap, devState);

        float random;
        genRandom(&random, devState);

        if (f > fr && f > fl) {}
        else if (f < fr && f < fl) {
            heading += (2 * (random - .5)) * config->RA;
        }
        else if (fl < fr) {
            heading -= random * config->RA;
        }
        else if (fr < fl) {
            heading += random * config->RA;
        }

        move(trailMap, devState);
    }
};

particle* globParticles;
float* diffusedTrailMap;

__global__
void updateParticles(particle *particles, float *trailMap, curandState *devState, modelConfig *mdlCfg)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < mdlCfg->numParticles; i += stride) {
        particles[i].update(trailMap, devState);
    }
}

__global__
void diffuseDecay(float *trailMap, float *diffusedTrailMap, modelConfig *mdlCfg) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % mdlCfg->width;
    int j = index / mdlCfg->width;

    int sum[3] = { 0, 0, 0 };
    float originalValue[3];

    for (int k = 0; k < 3; k++) {
        originalValue[k] = get(trailMap, i, j, k, mdlCfg);
    }

    for (int offsetX = -mdlCfg->diffK; offsetX <= mdlCfg->diffK; offsetX++) {
        for (int offsetY = -mdlCfg->diffK; offsetY <= mdlCfg->diffK; offsetY++) {

            int sampleX = MIN(mdlCfg->width - 1, MAX(0, i + offsetX));
            int sampleY = MIN(mdlCfg->height - 1, MAX(0, j + offsetY));

            for (int k = 0; k < 3; k++) {
                sum[k] += get(trailMap, sampleX, sampleY, k, mdlCfg);
            }
        }
    }

    float blur, diff, evap;
    for (int k = 0; k < 3; k++) {
        blur = sum[k] / ((mdlCfg->diffK * 2 + 1) * (mdlCfg->diffK * 2 + 1));
        diff = blur * mdlCfg->diffuseSpeed + originalValue[k] * (1 - mdlCfg->diffuseSpeed);
        evap = MAX(0, diff - mdlCfg->decayT);
        set(evap, diffusedTrailMap, i, j, k, mdlCfg);
    }
}

__global__
void initParticles(particle* particles, curandState* devState, particleConfig *cfg, modelConfig *mdlCfg) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < mdlCfg->numParticles; i += stride) {
        particles[i].initialize(index, devState, cfg, mdlCfg);
    }
}

extern "C"
void initCuda()
{
    // Initialize model config to defaults
    checkCudaErrors(cudaMallocManaged(&globMdlCfg, sizeof(modelConfig)));
    globMdlCfg->width = 1024;
    globMdlCfg->height = 1024;
    globMdlCfg->percentAreaPop = (float)rand() / RAND_MAX * 2;
    globMdlCfg->numParticles = 1000000;
    globMdlCfg->numSpecies = rand() % 3 + 1;
    globMdlCfg->start = rand() % 2;
    globMdlCfg->diffK = 1;
    globMdlCfg->diffuseSpeed = (float) rand() / RAND_MAX * .5 + .25;
    globMdlCfg->decayT = (float) rand() / RAND_MAX * .2;

    printf("Model Parameters:\n");
    printf("Width: %d, Height: %d, numParticles: %d, numSpecies: %d, Start: %d\n", globMdlCfg->width, globMdlCfg->height, globMdlCfg->numParticles, globMdlCfg->numSpecies, globMdlCfg->start);
    printf("decayT: %f, diffK: %d, diffuseSpeed %f\n", globMdlCfg->decayT, globMdlCfg->diffK, globMdlCfg->diffuseSpeed);

    // Initialize particle configs
    checkCudaErrors(cudaMallocManaged(&globCfg, globMdlCfg->numSpecies * sizeof(particleConfig)));
    for (int i = 0; i < globMdlCfg->numSpecies; i++) {
        globCfg[i].depositT = (float) rand() / RAND_MAX * .5 + .5;
        globCfg[i].RA = (float) rand() / RAND_MAX * M_PI / 2;
        globCfg[i].SA = (float) rand() / RAND_MAX * M_PI / 2;
        globCfg[i].SO = rand() % 10 + 1;
        globCfg[i].SS = rand() % 2 + 1;
        globCfg[i].SW = rand() % 2 + 1;
        globCfg[i].PCD = 0; // (float)rand() / RAND_MAX * .1;
        if (globMdlCfg->numSpecies == 1) {
            globCfg[i].index = -1;
        }
        else {
            globCfg[i].index = i;
        }

        printf("Species %d: \n", i);
        printf("depositT: %f, RA: %f, SA: %f, SO: %d, SS: %d, SW: %d, PCD: %f\n", globCfg[i].depositT, globCfg[i].RA, globCfg[i].SA, globCfg[i].SO, globCfg[i].SS, globCfg[i].SW, globCfg[i].PCD);
    }
    
    particleNumBlocks = (globMdlCfg->numParticles + blockSize - 1) / blockSize;
    trailMapNumBlocks = (globMdlCfg->width * globMdlCfg->height + blockSize - 1) / blockSize;

    printf("blockSize: %d, particleNumBlocks: %d, trailMapNumBlocks: %d\n", blockSize, particleNumBlocks, trailMapNumBlocks);

    // Initialize curandState for each particles thread
    checkCudaErrors(cudaMallocManaged(&devState, particleNumBlocks * blockSize * sizeof(curandState)));
    initCurand << <particleNumBlocks, blockSize >> > (clock(), devState);
    getLastCudaError("initCurand failed");

    cudaDeviceSynchronize();

    // Initialize particles
    checkCudaErrors(cudaMallocManaged(&globParticles, globMdlCfg->numParticles * sizeof(particle)));

    initParticles << <particleNumBlocks, blockSize >> > (globParticles, devState, globCfg, globMdlCfg);
    getLastCudaError("initParticles failed");

    cudaDeviceSynchronize();

    // Initialize secondary trail map

    checkCudaErrors(cudaMallocManaged(&diffusedTrailMap, globMdlCfg->width * globMdlCfg->height * 3 * sizeof(float)));
}

extern "C"
void render_kernel(float *d_output)
{
    updateParticles << <particleNumBlocks, blockSize >> > (globParticles, d_output, devState, globMdlCfg);
    getLastCudaError("updateParticles failed");

    diffuseDecay << <trailMapNumBlocks, blockSize >> > (d_output, diffusedTrailMap, globMdlCfg);
    getLastCudaError("diffuseDecay failed");
    
    cudaDeviceSynchronize();

    cudaMemcpy(d_output, diffusedTrailMap, globMdlCfg->width * globMdlCfg->height * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    getLastCudaError("memcpy failed");
}

extern "C"
void randomizeHeadings() {
    for (int i = 0; i < globMdlCfg->numParticles; i++) {
        globParticles[i].heading = (float)rand() / RAND_MAX * 2 * M_PI;
    }
}

extern "C"
void randomizeHeadingsHalf() {
    for (int i = 0; i < globMdlCfg->numParticles / 2; i++) {
        globParticles[i].heading = (float)rand() / RAND_MAX * 2 * M_PI;
    }
}

extern "C"
modelConfig * getModelConfig() {
    modelConfig* hostModelConfig = (modelConfig*)malloc(sizeof(modelConfig));

    hostModelConfig->width = globMdlCfg->width;
    hostModelConfig->height = globMdlCfg->height;
    hostModelConfig->numParticles = globMdlCfg->numParticles;
    hostModelConfig->numSpecies = globMdlCfg->numSpecies;
    hostModelConfig->start = globMdlCfg->start;
    hostModelConfig->diffK = globMdlCfg->diffK;
    hostModelConfig->diffuseSpeed = globMdlCfg->diffuseSpeed;
    hostModelConfig->decayT = globMdlCfg->decayT;

    return hostModelConfig;
}

extern "C"
void updatetModelConfig(modelConfig * mdlCfg) {
    globMdlCfg->width = mdlCfg->width;
    globMdlCfg->height = mdlCfg->height;
    globMdlCfg->numParticles = mdlCfg->numParticles;
    globMdlCfg->numSpecies = mdlCfg->numSpecies;
    globMdlCfg->start = mdlCfg->start;
    globMdlCfg->diffK = mdlCfg->diffK;
    globMdlCfg->diffuseSpeed = mdlCfg->diffuseSpeed;
    globMdlCfg->decayT = mdlCfg->decayT;
}

extern "C"
particleConfig * getParticleConfig(int id) {
    particleConfig* hostParticleConfig = (particleConfig*)malloc(sizeof(particleConfig));

    hostParticleConfig[id].depositT = globCfg->depositT;
    hostParticleConfig[id].RA = globCfg->RA;
    hostParticleConfig[id].SA = globCfg->SA;
    hostParticleConfig[id].SO = globCfg->SO;
    hostParticleConfig[id].SS = globCfg->SS;
    hostParticleConfig[id].SW = globCfg->SW;

    return hostParticleConfig;
}

extern "C"
void updateParticleConfig(int id, particleConfig * cfg) {
    if (id < globMdlCfg->numSpecies) {
        globCfg[id].depositT = cfg->depositT;
        globCfg[id].RA = cfg->RA;
        globCfg[id].SA = cfg->SA;
        globCfg[id].SO = cfg->SO;
        globCfg[id].SS = cfg->SS;
        globCfg[id].SW = cfg->SW;

        printf("Species %d New Parameters:\n", id);
        printf("depositT: %f, RA: %f, SA: %f, SO: %d, SS: %d, SW: %d\n", globCfg[id].depositT, globCfg[id].RA, globCfg[id].SA, globCfg[id].SO, globCfg[id].SS, globCfg[id].SW);
    }
    else {
        printf("Index %d out of range\n", id);
    }
}

extern
void cleanupCuda()
{
    cudaFree(devState);
    cudaFree(globParticles);
}