#ifndef __riscv
#include "Tinn.h"
#include "helper.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// Learns and predicts hand written digits with 98% accuracy.
int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
    const int nips = 256;
    const int nops = 10;
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate.
    // The number of iterations can be changed for stronger training.
    float rate = 1.0f;
    const int nhid = 28;
    const float anneal = 0.99f;
    const int iterations = 128;
    // Load the training set.
    const Data data = build("semeion.data", nips, nops);
    // Train, baby, train.
    const Tinn tinn = xtbuild(nips, nhid, nops);
    for(int i = 0; i < iterations; i++)
    {
        shuffle(data);
        float error = 0.0f;
        for(int j = 0; j < data.rows; j++)
        {
            const float* const in = data.in[j];
            const float* const tg = data.tg[j];
            error += xttrain(tinn, in, tg, rate);
        }
        printf("error %.12f :: learning rate %f\n",
            (double) error / data.rows,
            (double) rate);
        rate *= anneal;
    }
    // This is how you save the neural network to disk.
    xtsave(tinn, "saved.tinn");
    xtfree(tinn);
    dfree(data);
    return 0;
}
#endif