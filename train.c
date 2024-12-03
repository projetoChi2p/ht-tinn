#ifndef __riscv
#include "Tinn.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Data object.
typedef struct
{
    // 2D floating point array of input.
    float** in;
    // 2D floating point array of target.
    float** tg;
    // Number of inputs to neural network.
    int nips;
    // Number of outputs to neural network.
    int nops;
    // Number of rows in file (number of sets for neural network).
    int rows;
}
Data;

// Returns the number of lines in a file.
static int lns(FILE* const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Reads a line from a file.
static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

// New data object.
static Data ndata(const int nips, const int nops, const int rows)
{
    // 2D floating point array of input.
    float** in = new2d(rows, nips);
    printf("Made in\n");
    // 2D floating point array of target.
    float** tg = new2d(rows, nops);
    printf("Made tg\n");
    const Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char* line, const int row)
{
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

static void parse_mnist(const Data data, char* line, const int row)
{
    const int cols = data.nips + 1;
    const int tgt = atoi(strtok(line, ","));
    for (int i = 0; i < data.nops; i++)
    {
        data.tg[row][i] = i == tgt ? 1.0f : 0.0f;
    }

    for (int col = 1; col < cols; col++){
        const int val = atoi(strtok(NULL, ","));
        data.in[row][col] = val / 255.0f;
    }
}

// Frees a data object from the heap.
void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

// Randomly shuffles a data object.
void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float* ot = d.tg[a];
        float* it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
Data build(const char* path, const int nips, const int nops, const int mnist)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get it from the machine learning database: ");
        printf("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        printf("Or from the MNIST database: ");
        printf("https://www.kaggle.com/datasets/oddrationale/mnist-in-csv\n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(nips, nops, rows);
    if (mnist != 0){
        // Skip the first line
        char* line = readln(file);
        (void) line;
    }
    for(int row = mnist == 0 ? 0 : 1; row < rows; row++)
    {
        char* line = readln(file);
        if (mnist != 0){
            parse_mnist(data, line, row);
        } else {
            parse(data, line, row);
        }
        free(line);
    }
    fclose(file);
    return data;
}

// Input and output size is harded coded here as machine learning
// repositories usually don't include the input and output size in the data itself.
// Hyper Parameters.
// Learning rate is annealed and thus not constant.
// It can be fine tuned along with the number of hidden layers.
// Feel free to modify the anneal rate.
// The number of iterations can be changed for stronger training.
int train(const int nips, const int nops, const int nhid, float learn_rate, const float anneal, const int iterations, const int mnist)
{
    const Data data = mnist == 0 ? build("semeion.data", nips, nops, 0) : build("mnist_train.csv", nips, nops, 1);
    // Load the training set.

    // Train, baby, train.
    const Tinn tinn = xtbuild(nips, nhid, nops);

    for (int i = 0; i < iterations; i++)
    {
        shuffle(data);
        float error = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            const float *const in = data.in[j];
            const float *const tg = data.tg[j];
            error += xttrain(tinn, in, tg, learn_rate);
        }
        printf("error %.12f :: learning rate %f\n",
               (double)error / data.rows,
               (double)learn_rate);
        learn_rate *= anneal;
    }
    // This is how you save the neural network to disk.
    printf("Saving neural network\n");
    char *path = mnist == 0 ? "semeion.tinn" : "mnist.tinn";
    xtsave(tinn, path);
    printf("Neural network saved\n");
    // All done. Let's clean up.
    xtfree(tinn);
    dfree(data);
    return 0;
}


// Learns and predicts hand written digits with 98% accuracy.
int main(int argc, char **argv)
{
    int mnist = argc == 1 ? 0 : atoi(argv[1]);
    printf("Training for: %s", mnist ? "MNIST\n" : "SEMEION\n");
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Hyper Parameters.
    const int nips = mnist == 0 ? 256 : 784;
    const int nops = 10;
    float rate = 1.0f;
    const int nhid = mnist == 0 ? 28 : 196;
    const float anneal = 0.99f;
    const int iterations = 128;
    return train(nips, nops, nhid, rate, anneal, iterations, mnist);
}
#endif