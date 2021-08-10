#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>

#include "config.h"

const char* sSDKsample = "physarum";

float w = 0.5;  // texture coordinate in z

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

bool animate = true;
bool fullscreen = false;

StopWatchInterface* timer = NULL;
modelConfig* mdlCfg;
float zoomFactor = 1.0;

float* d_output = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
volatile int g_GraphicsMapFlag = 0;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

extern     void cleanupCuda();
extern "C" void render_kernel(float *trailMap);
extern "C" void initCuda();
extern "C" modelConfig *getModelConfig();
extern "C" void updatetModelConfig(modelConfig * mdlCfg);
extern "C" particleConfig * getParticleConfig(int id);
extern "C" void updateParticleConfig(int id, particleConfig * cfg);
extern "C" void randomizeHeadings();
extern "C" void randomizeHeadingsHalf();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "%s: %3.1f fps", sSDKsample, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = ftoi(MAX(1.0f, ifps));
        sdkResetTimer(&timer);
    }
}

// render image using CUDA
void render()
{
    // map PBO to get CUDA device pointer
    g_GraphicsMapFlag++;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));
    // printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // call CUDA kernel, writing results to PBO
    render_kernel(d_output);

    if (g_GraphicsMapFlag)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        g_GraphicsMapFlag--;
    }
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(mdlCfg->width, mdlCfg->height, GL_RGB, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glutReportErrors();

    sdkStopTimer(&timer);
    computeFPS();
}

void idle()
{
    if (animate)
    {
        w += 0.01f;
        glutPostRedisplay();
    }
}

void mouse(int button, int state, int mx, int my)
{
    float x = .5 - (float) mx / mdlCfg->width;
    float y = .5 - (float) my / mdlCfg->height;

    printf("%d %d %f %f\n", mx, my, x, y);

    if (button == 3) {
        zoomFactor += .1;
    }
    else if (button == 4) {
        zoomFactor -= .1;
    }
    
    glPixelZoom(zoomFactor, zoomFactor);

}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'f':
        if (fullscreen) {
            glutPositionWindow(0, 0);
        }
        else {
            glutFullScreen();
        }
        fullscreen = !fullscreen;
        break;
    case 's':
        cleanupCuda();
        initCuda();
        mdlCfg = getModelConfig();
        break;
    case 'r':
        randomizeHeadings();
        break;
    case 'h':
        randomizeHeadingsHalf();
        break;
    default:
        break;
    }
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    // add extra check to unmap the resource before unregistering it
    if (g_GraphicsMapFlag)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        g_GraphicsMapFlag--;
    }

    // unregister this buffer object from CUDA C
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1, &pbo);
    cleanupCuda();
}

void initGLBuffers()
{
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, mdlCfg->width * mdlCfg->height * 3 * sizeof(GL_FLOAT), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

void initGL(int* argc, char** argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(mdlCfg->width, mdlCfg->height);
    glutCreateWindow("Physarum");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    srand(time(NULL));

    printf("%s Starting...\n\n", sSDKsample);

    initCuda();

    mdlCfg = getModelConfig();

    initGL(&argc, argv);

    // OpenGL buffers
    initGLBuffers();

    glutCloseFunc(cleanup);

    glutMainLoop();

    exit(EXIT_SUCCESS);
}
