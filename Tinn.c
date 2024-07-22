#include "Tinn.h"

#ifndef TINN_EXP
#include <math.h>
#define TINN_EXP(x) expf(x)
#endif

#ifndef TINN_ATOF
#include <math.h>
#define TINN_ATOF(x) atof(x)
#endif

#ifdef FREERTOS
#include <ff_headers.h>
#include <ff_stdio.h>
#include <FreeRTOS.h>

#ifndef TINN_RANDOM
#define TINN_RAND() rand()
#endif

#ifndef TINN_FREE
#define TINN_FREE(x) vPortFree(x)
#endif

#ifndef TINN_CALLOC
#define TINN_CALLOC(n, s) pvPortCalloc(n, s)
#endif

#ifndef TINN_PRINT
#include "printf.h"
#define TINN_PRINT printf
#endif

#ifndef TINN_FOPEN
#define TINN_FOPEN(path, mode) ff_fopen(path, mode)
#endif

#ifndef TINN_FCLOSE
#define TINN_FCLOSE(file) ff_fclose(file)
#endif

#ifndef TINN_FPRINTF
#define TINN_FPRINTF ff_fprintf
#endif

#ifndef TINN_FGETC
#define TINN_FGETC ff_fgetc
#endif

#ifndef TINN_FGETS
#define TINN_FGETS ff_fgets
#endif

#ifndef TINN_FILE
#define TINN_FILE FF_FILE
#endif

#else

#ifndef TINN_RAND
#include <stdlib.h>
#define TINN_RAND() rand()
#endif

#ifndef TINN_FREE
#include <stdio.h>
#define TINN_FREE(x) free(x)
#endif

#ifndef TINN_CALLOC
#include <stdio.h>
#define TINN_CALLOC(n, s) calloc(n, s)
#endif

#ifndef TINN_PRINT
#include <stdio.h>
#define TINN_PRINT printf
#endif

#ifndef TINN_FOPEN
#include <stdio.h>
#define TINN_FOPEN(path, mode) fopen(path, mode)
#endif

#ifndef TINN_FCLOSE
#include <stdio.h>
#define TINN_FCLOSE(file) fclose(file)
#endif

#ifndef TINN_FPRINTF
#include <stdio.h>
#define TINN_FPRINTF fprintf
#endif

#ifndef TINN_FGETC
#include <stdio.h>
#define TINN_FGETC fgetc
#endif

#ifndef TINN_FGETS
#include <stdio.h>
#define TINN_FGETS fgets
#endif

#ifndef TINN_FILE
#include <stdio.h>
#define TINN_FILE FILE
#endif

#endif

static int xtGetInt(TINN_FILE *file, char delim)
{
    int n = 0;
    char c;
    while ((c = TINN_FGETC(file)) != delim)
        n = n * 10 + (c - '0');
    return n;
}

static float xtGetNextFloat(TINN_FILE *file)
{
    char buffer[256];
    if (TINN_FGETS(buffer, 256, file) == NULL)
    {
        return 0;
    }

    float n = TINN_ATOF(buffer);

    return n;
}

// Computes error.
static float err(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Returns partial derivative of error function.
static float pderr(const float a, const float b)
{
    return a - b;
}

// Computes total error of target to output.
static float toterr(const float *const tg, const float *const o, const int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
        sum += err(tg[i], o[i]);
    return sum;
}

// Activation function.
static float act(const float a)
{
    return 1.0f / (1.0f + TINN_EXP(-a));
}

// Returns partial derivative of activation function.
static float pdact(const float a)
{
    return a * (1.0f - a);
}

// Returns floating point random from 0.0 - 1.0.
static float frand()
{
    return TINN_RAND() / (float)RAND_MAX;
}

// Performs back propagation.
static void bprop(const Tinn t, const float *const in, const float *const tg, float rate)
{
    for (int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        // Calculate total error change with respect to output.
        for (int j = 0; j < t.nops; j++)
        {
            const float a = pderr(t.o[j], tg[j]);
            const float b = pdact(t.o[j]);
            sum += a * b * t.x[j * t.nhid + i];
            // Correct weights in hidden to output layer.
            t.x[j * t.nhid + i] -= rate * a * b * t.h[i];
        }
        // Correct weights in input to hidden layer.
        for (int j = 0; j < t.nips; j++)
            t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * in[j];
    }
}

// Performs forward propagation.
static void fprop(const Tinn t, const float *const in)
{
    // Calculate hidden layer neuron values.
    for (int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for (int i = 0; i < t.nops; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = act(sum + t.b[1]);
    }
}

// Randomizes tinn weights and biases.
static void wbrand(const Tinn t)
{
    for (int i = 0; i < t.nw; i++)
        t.w[i] = frand() - 0.5f;
    for (int i = 0; i < t.nb; i++)
        t.b[i] = frand() - 0.5f;
}

// Returns an output prediction given an input.
float *xtpredict(const Tinn t, const float *const in)
{
    fprop(t, in);
    return t.o;
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
float xttrain(const Tinn t, const float *const in, const float *const tg, float rate)
{
    fprop(t, in);
    bprop(t, in, tg, rate);
    return toterr(tg, t.o, t.nops);
}

// Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
Tinn xtbuild(const int nips, const int nhid, const int nops)
{
    Tinn t;
    // Tinn only supports one hidden layer so there are two biases.
    t.nb = 2;
    t.nw = nhid * (nips + nops);
    t.w = (float *)TINN_CALLOC(t.nw, sizeof(*t.w));
    t.x = t.w + nhid * nips;
    t.b = (float *)TINN_CALLOC(t.nb, sizeof(*t.b));
    t.h = (float *)TINN_CALLOC(nhid, sizeof(*t.h));
    t.o = (float *)TINN_CALLOC(nops, sizeof(*t.o));
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    wbrand(t);
    return t;
}

// Saves a tinn to disk.
void xtsave(const Tinn t, const char *const path)
{
    TINN_FILE *const file = TINN_FOPEN(path, "w");
    // Save header.
    TINN_FPRINTF(file, "%d %d %d\n", t.nips, t.nhid, t.nops);
    // Save biases and weights.
    for (int i = 0; i < t.nb; i++)
        TINN_FPRINTF(file, "%f\n", (double)t.b[i]);
    for (int i = 0; i < t.nw; i++)
        TINN_FPRINTF(file, "%f\n", (double)t.w[i]);
    TINN_FCLOSE(file);
}

// Loads a tinn from disk.
Tinn xtload(const char *const path)
{
    TINN_FILE *const file = TINN_FOPEN(path, "r");
    if (file == NULL)
        return (Tinn){0};
    // Load header with number of inputs, hidden neurons, and outputs.
    // this must be done with fread because fscanf is not supported in freertos.
    // TINN_FSCANF(file, "%d %d %d\n", &nips, &nhid, &nops);
    int nips = xtGetInt(file, ' ');
    int nhid = xtGetInt(file, ' ');
    int nops = xtGetInt(file, '\n');
    // Build a new tinn.
    const Tinn t = xtbuild(nips, nhid, nops);
    // Load bias and weights.
    for (int i = 0; i < t.nb; i++)
        t.b[i] = xtGetNextFloat(file);
    for (int i = 0; i < t.nw; i++)
        t.w[i] = xtGetNextFloat(file);
    TINN_FCLOSE(file);
    return t;
}

// Frees object from heap.
void xtfree(const Tinn t)
{
    TINN_FREE(t.w);
    TINN_FREE(t.b);
    TINN_FREE(t.h);
    TINN_FREE(t.o);
}

// Prints an array of floats. Useful for printing predictions.
void xtprint(const float *arr, const int size)
{
    for (int i = 0; i < size; i++)
        TINN_PRINT("%f ", (double)arr[i]);
    TINN_PRINT("\n");
}
