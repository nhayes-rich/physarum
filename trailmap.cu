#include "trailmap.cuh"

__device__
float trailMap::get(int i, int j, int k, modelConfig* mdlCfg) {
    return map[i * mdlCfg->width * 3 + j * 3 + 3 - k];
}

__device__
void trailMap::set(float val, int i, int j, int k, modelConfig* mdlCfg) {
    map[i * mdlCfg->width * 3 + j * 3 + 3 - k] = val;
}