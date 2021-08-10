#include <helper_cuda.h>
#include <helper_math.h>

#include "config.h"

#ifndef __TRAILMAP__
#define __TRAILMAP__

// 3d to 1d index
class trailMap {
public:
	float* map;
	
	__device__
	float get(int i, int j, int k, modelConfig* mdlCfg);

	__device__
	void set(float val, int i, int j, int k, modelConfig* mdlCfg);
};

#endif