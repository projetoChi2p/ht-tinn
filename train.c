#ifndef __riscv
#include "Tinn.h"
#include "helper.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

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

#define MNIST

// Learns and predicts hand written digits with 98% accuracy.
int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
#ifndef MNIST
    const int nips = 256;
    const int nops = 10;

    float rate = 1.0f;
    const int nhid = 28;
    const float anneal = 0.99f;
    const int iterations = 128;
    const int mnist = 0;
#else
    const int nips = 784;
    const int nops = 10;

    float rate = 1.0f;
    const int nhid = 196;
    const float anneal = 0.99f;
    const int iterations = 128;
    const int mnist = 1;
#endif
    return train(nips, nops, nhid, rate, anneal, iterations, mnist);
}
#endif