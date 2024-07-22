#include "Tinn.h"
#include "helper.h"
#include <time.h>
#include <stdlib.h>

int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).
    const int nops = 10;; // Number of outputs to neural network.
    const int nips = 256;
    // Load the training set.
    const Data data = build("semeion.data", nips, nops);
    shuffle(data);
    const float* const in = data.in[0];
    const float* const tg = data.tg[0];

    // This is how you load the neural network from disk.
    const Tinn loaded = xtload("saved.tinn");
    const float *const pd = xtpredict(loaded, in);
    // Prints target.
    xtprint(tg, nops);
    // Prints prediction.
    xtprint(pd, nops);
    // All done. Let's clean up.
    xtfree(loaded);
    return 0;
}