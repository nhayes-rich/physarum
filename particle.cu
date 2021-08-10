#include "particle.cuh"

// Particles

__device__
void particle::initialize(int seed, curandState* devState, particleConfig* cfg, modelConfig* mdlCfg) {
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
float particle::sense(float angleOffset, trailMap tm, curandState* devState) {
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
                    sum += tm.get(pos[0], pos[1], k, mdlConfig) * mask[k];
                }
            }
        }
    }
    return sum;
}

__device__
void particle::move(trailMap tm, curandState* devState) {
    float cosHeading = cos(heading);
    float sinHeading = sin(heading);

    float newX = config->SS * cosHeading + location[0];
    float newY = config->SS * sinHeading + location[1];

    int indX = (int)floor(newX);
    int indY = (int)floor(newY);

    while (indX < 0 || indX >= mdlConfig->width || indY < 0 || indY >= mdlConfig->height) {
        float random;
        genRandom(&random, devState);
        heading += random * 2 * M_PI;

        cosHeading = cos(heading);
        sinHeading = sin(heading);

        newX = config->SS * cosHeading + location[0];
        newY = config->SS * sinHeading + location[1];

        indX = (int)floor(newX);
        indY = (int)floor(newY);
    }

    location[0] = newX, location[1] = newY;

    if (config->index == -1) {
        tm.set(config->depositT, indX, indY, 0, mdlConfig);
        tm.set(config->depositT, indX, indY, 1, mdlConfig);
        tm.set(config->depositT, indX, indY, 2, mdlConfig);
    }                         
    else {                    
        tm.set(config->depositT, indX, indY, config->index, mdlConfig);
    }


}

__device__
void particle::update(trailMap tm, curandState* devState) {
    move(tm, devState);

    float f = sense(0, tm, devState);
    float fl = sense(config->SA, tm, devState);
    float fr = sense(-config->SA, tm, devState);

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
}
