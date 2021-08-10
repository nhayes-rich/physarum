#ifndef __CONFIG__
#define __CONFIG__

typedef unsigned char byte;

enum start { CENTER, CIRCLE };

struct modelConfig {
	int width;
	int height;
	int start;
	float percentAreaPop;
	int numParticles;
	int numSpecies;
	int diffK;
	float decayT;
	float diffuseSpeed;
};

struct particleConfig {
	float SA;
	float RA;
	int SO;
	int SW;
	int SS;
	float depositT;
	int index;
	float PCD;
	float aversion;
};

#endif