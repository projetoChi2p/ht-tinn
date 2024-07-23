#pragma once
#ifndef __riscv
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

// Randomly shuffles a data object.
void shuffle(const Data d);

// Frees a data object from the heap.
void dfree(const Data d);

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
Data build(const char* path, const int nips, const int nops);
#endif