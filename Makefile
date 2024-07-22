TRAIN = train
PREDICT = predict

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

LDFLAGS = -lm

CC = gcc

TINN_SRC = Tinn.c

TRAIN_SRC = train.c helper.c

PREDICT_SRC = predict.c helper.c

all: train predict

train: $(TINN_SRC) $(TRAIN_SRC)
	$(CC) -o $(TRAIN) $(TINN_SRC) $(TRAIN_SRC) $(CFLAGS) $(LDFLAGS)

predict: $(TINN_SRC) $(PREDICT_SRC)
	$(CC) -o $(PREDICT) $(TINN_SRC) $(PREDICT_SRC) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TRAIN) $(PREDICT)
