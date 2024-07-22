#include "Tinn.h"

#ifndef PREDICT_ATOF
#include <math.h>
#define PREDICT_ATOF(x) atof(x)
#endif

#ifdef FREERTOS
#include <ff_headers.h>
#include <ff_stdio.h>
#include <FreeRTOS.h>

#ifndef PREDICT_RAND
#define PREDICT_RAND() rand()
#endif

#ifndef PREDICT_FREE
#define PREDICT_FREE(x) vPortFree(x)
#endif

#ifndef PREDICT_MALLOC
#define PREDICT_MALLOC(n) pvPortMalloc(n)
#endif

#ifndef PREDICT_REALLOC
#define PREDICT_REALLOC(n, s) pvPortRealloc(n, s)
#endif

#ifndef PREDICT_PRINT
#include "printf.h"
#define PREDICT_PRINT printf
#endif

#ifndef PREDICT_FOPEN
#define PREDICT_FOPEN(path, mode) ff_fopen(path, mode)
#endif

#ifndef PREDICT_FCLOSE
#define PREDICT_FCLOSE(file) ff_fclose(file)
#endif

#ifndef PREDICT_FGETC
#define PREDICT_FGETC ff_fgetc
#endif

#ifndef PREDICT_FILE
#define PREDICT_FILE FF_FILE
#endif

#ifndef PREDICT_REWIND
#include <stdio.h>
#define PREDICT_REWIND FF_rewind
#endif

#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PREDICT_RAND
#include <stdlib.h>
#define PREDICT_RAND() rand()
#endif

#ifndef PREDICT_FREE
#include <stdio.h>
#define PREDICT_FREE(x) free(x)
#endif

#ifndef PREDICT_MALLOC
#include <stdio.h>
#define PREDICT_MALLOC(n) malloc(n)
#endif

#ifndef PREDICT_REALLOC
#include <stdio.h>
#define PREDICT_REALLOC(n, s) realloc(n, s)
#endif

#ifndef PREDICT_PRINT
#include <stdio.h>
#define PREDICT_PRINT printf
#endif

#ifndef PREDICT_FOPEN
#include <stdio.h>
#define PREDICT_FOPEN(path, mode) fopen(path, mode)
#endif

#ifndef PREDICT_FCLOSE
#include <stdio.h>
#define PREDICT_FCLOSE(file) fclose(file)
#endif

#ifndef PREDICT_FGETC
#include <stdio.h>
#define PREDICT_FGETC fgetc
#endif


#ifndef PREDICT_FILE
#include <stdio.h>
#define PREDICT_FILE FILE
#endif

#ifndef PREDICT_REWIND
#include <stdio.h>
#define PREDICT_REWIND rewind
#endif

#endif

// Returns the number of lines in a file.
static int lns(PREDICT_FILE *const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while ((ch = PREDICT_FGETC(file)) != EOF)
    {
        if (ch == '\n')
            lines++;
        pc = ch;
    }
    if (pc != '\n')
        lines++;
    PREDICT_REWIND(file);
    return lines;
}

// Reads a line from a file.
static char *readln(PREDICT_FILE *const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char *line = (char *)PREDICT_MALLOC((size) * sizeof(char));
    while ((ch = PREDICT_FGETC(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if (reads + 1 == size)
            line = (char *)PREDICT_REALLOC((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

typedef struct
{
    float *in;
    float *tg;
} Data;

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char *line, const int nips, const int nops)
{
    const int cols = nips + nops;
    for (int col = 0; col < cols; col++)
    {
        const float val = PREDICT_ATOF(strtok(col == 0 ? line : NULL, " "));
        if (col < nips)
            data.in[col] = val;
        else
            data.tg[col - nips] = val;
    }
}

Data load_row(const char *path, const int nips, const int nops)
{
    PREDICT_FILE *file = PREDICT_FOPEN(path, "r");
    if (file == NULL)
    {
        PREDICT_PRINT("Error: Could not open file %s\n", path);
        return (Data){NULL, NULL};
    }
    const int rows = lns(file);
    Data data = {
        (float *)PREDICT_MALLOC(nips * sizeof(float)),
        (float *)PREDICT_MALLOC(nops * sizeof(float))};

    const int row = PREDICT_RAND() % rows;
    char *line;
    for (int i = 0; i < row; i++)
    {
        line = readln(file);
        PREDICT_FREE(line);
    }
    line = readln(file);
    parse(data, line, nips, nops);
    PREDICT_FREE(line);
    PREDICT_FCLOSE(file);
    return data;
}

int predict(int seed)
{
    // Tinn does not seed the random number generator.
    srand(seed);
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).
    const int nops = 10;
    ; // Number of outputs to neural network.
    const int nips = 256;
    // Load the training set.
    const Data data = load_row("semeion.data", nips, nops);
    const float *const in = data.in;
    const float *const tg = data.tg;
    if (in == NULL || tg == NULL)
        return 1;

    // This is how you load the neural network from disk.
    const Tinn loaded = xtload("saved.tinn");
    const float *const pd = xtpredict(loaded, in);
    // Prints target.
    xtprint(tg, nops);
    // Prints prediction.
    xtprint(pd, nops);
    // All done. Let's clean up.
    xtfree(loaded);
    PREDICT_FREE(data.in);
    PREDICT_FREE(data.tg);
    return 0;
}

#ifndef __riscv
#include <time.h>
int main()
{
    return predict(time(0));
}
#endif