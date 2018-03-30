#include "Tinn.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct
{
    double** in;
    double** tg;
    int nips;
    int nops;
    int rows;
}
Data;

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

static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = ((char*) malloc((size) * sizeof(char)));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

static double** new2d(const int rows, const int cols)
{
    double** row = (double**) malloc((rows) * sizeof(double*));
    for(int r = 0; r < rows; r++)
        row[r] = (double*) malloc((cols) * sizeof(double));
    return row;
}

static Data ndata(const int nips, const int nops, const int rows)
{
    const Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

static void parse(const Data data, char* line, const int row)
{
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const double val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        double* ot = d.tg[a];
        double* it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

static Data build(const char* path, const int nips, const int nops)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get it from the machine learning database: ");
        printf("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(nips, nops, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

int main()
{
    // Input and output size is harded coded here,
    // so make sure the training data sizes match.
    const int nips = 256;
    const int nops = 10;
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    const int nhid = 32;
    double rate = 0.5;
    // Load the training set.
    const Data data = build("semeion.data", nips, nops);
    // Rock and roll.
    const Tinn tinn = xtbuild(nips, nhid, nops);
    for(int i = 0; i < 100; i++)
    {
        shuffle(data);
        double error = 0.0;
        for(int j = 0; j < data.rows; j++)
        {
            const double* const in = data.in[j];
            const double* const tg = data.tg[j];
            error += xttrain(tinn, in, tg, rate);
        }
        printf("error %.12f :: rate %f\n", error / data.rows, rate);
        rate *= 0.99;
    }
    // Ideally, you would load a testing set for predictions,
    // but for the sake of brevity the training set is reused.
    const double* const in = data.in[0];
    const double* const tg = data.tg[0];
    const double* const pd = xpredict(tinn, in);
    for(int i = 0; i < data.nops; i++) { printf("%f ", tg[i]); } printf("\n");
    for(int i = 0; i < data.nops; i++) { printf("%f ", pd[i]); } printf("\n");
    xtfree(tinn);
    dfree(data);
    return 0;
}