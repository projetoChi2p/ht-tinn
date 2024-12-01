#include "predict.h"
#include "Tinn.h"

#define MNIST

#ifndef PREDICT_ATOF
#include <math.h>
#define PREDICT_ATOF(x) atof(x)
#endif

#ifndef PREDICT_ATOI
#include <math.h>
#define PREDICT_ATOI(x) atoi(x)
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
#define PREDICT_REWIND ff_rewind
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
    PREDICT_PRINT("Counting lines\n");
    while ((ch = PREDICT_FGETC(file)) != EOF)
    {
        if (ch == '\n')
            lines++;
        pc = ch;
    }
    if (pc != '\n')
        lines++;
    PREDICT_PRINT("Rewinding file\n");
    PREDICT_REWIND(file);
    PREDICT_PRINT("File rewound\n");
    return lines;
}

// Reads a line from a file.
static char *readln(PREDICT_FILE *const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    //PREDICT_PRINT("Reading line\n");
    char *line = (char *)PREDICT_MALLOC((size) * sizeof(char));
    while ((ch = PREDICT_FGETC(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if (reads + 1 == size){
            int new_size = size * 2;
            char *new_line = (char *)PREDICT_MALLOC(new_size * sizeof(char));
            for (int i = 0; i < size; i++)
            {
                new_line[i] = line[i];
            }
            PREDICT_FREE(line);
            line = new_line;
            size = new_size;
        }
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
#ifndef MNIST
    const int cols = nips + nops;
    for (int col = 0; col < cols; col++)
    {
        const float val = PREDICT_ATOF(strtok(col == 0 ? line : NULL, " "));
        if (col < nips)
            data.in[col] = val;
        else
            data.tg[col - nips] = val;
    }
#else
    const int cols = nips + 1;
    const int tgt = PREDICT_ATOI(strtok(line, ","));
    for (int i = 0; i < nops; i++)
    {
        data.tg[i] = i == tgt ? 1.0f : 0.0f;
    }
    for (int col = 1; col < cols; col++)
    {
        const int val = PREDICT_ATOI(strtok(NULL, ","));
        data.in[col - 1] = val / 255.0f;
    }
#endif
}

Data load_row(const char *path, const int nips, const int nops)
{
    PREDICT_PRINT("Opening file\n");
    PREDICT_FILE *file = PREDICT_FOPEN(path, "r");
    if (file == NULL)
    {
        PREDICT_PRINT("Error: Could not open file %s\n", path);
        return (Data){NULL, NULL};
    }
    PREDICT_PRINT("File opened\n");
    const int rows = lns(file);
    PREDICT_PRINT("Rows counted: %d\n", rows);
    Data data = {
        (float *)PREDICT_MALLOC(nips * sizeof(float)),
        (float *)PREDICT_MALLOC(nops * sizeof(float))};

    const int row = PREDICT_RAND() % rows;
    PREDICT_PRINT("Random row selected: %d\n", row);
    char *line;
    for (int i = 0; i < row; i++)
    {
        line = readln(file);
        PREDICT_FREE(line);
    }
    PREDICT_PRINT("Reading line\n");
    line = readln(file);
    PREDICT_PRINT("Line read\n");
    parse(data, line, nips, nops);
    PREDICT_PRINT("Data parsed\n");
    PREDICT_FREE(line);
    PREDICT_FCLOSE(file);
    return data;
}

#include <stdint.h>

int predict(int seed)
{
    // Tinn does not seed the random number generator.
    srand(seed);
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).
#ifndef MNIST
    const int nops = 10;
    // Number of outputs to neural network.
    const int nips = 256;
    PREDICT_PRINT("Loading data\n");
    const Data data = load_row("semeion.data", nips, nops);
#else
    const int nops = 10;
    // Number of outputs to neural network.
    const int nips = 784;
    const Data data = load_row("mnist_test.csv", nips, nops);
#endif
    // Load the training set.
    const float *const in = data.in;
    const float *const tg = data.tg;
    if (in == NULL || tg == NULL){
        PREDICT_PRINT("Error: Could not load data\n");
        return 1;
    }

    // This is how you load the neural network from disk.
    PREDICT_PRINT("Loading neural network\n");
#ifndef MNIST
    const Tinn loaded = xtload("saved.tinn");
#else
    const Tinn loaded = xtload("mnist.tinn");
#endif
    PREDICT_PRINT("Predicting\n");
#ifdef FREERTOS
    uint32_t start = xTaskGetTickCount();
#else
    uint32_t start = 0;
#endif
    const float *const pd = xtpredict(loaded, in);
#ifdef FREERTOS
    uint32_t end = xTaskGetTickCount();
#else
    uint32_t end = 0;
#endif
    // Prints target.
    xtprint(tg, nops);
    // Prints prediction.
    xtprint(pd, nops);
    // All done. Let's clean up.
    xtfree(loaded);
    PREDICT_FREE(data.in);
    PREDICT_FREE(data.tg);
    return end - start;
}

#ifndef __riscv
#include <time.h>
int main(int argc, char **argv)
{
    if (argc == 1)
    {
        return predict(time(0));
    }
    
    int count = atoi(argv[1]);
    for (int i = 0; i < count; i++)
    {
        if (predict(time(0)) != 0)
        {
            return 1;
        }
    }

    return 0;
}
#endif