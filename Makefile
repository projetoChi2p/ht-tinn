TRAIN = train
PREDICT = predict
ifeq ($(OS),Windows_NT)
    EXT = .exe
endif

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

LDFLAGS = -lm

CC = gcc

TINN_SRC = Tinn.c

TRAIN_SRC = train.c

PREDICT_SRC = predict.c

all: train predict

train: $(TINN_SRC) $(TRAIN_SRC)
	$(CC) -o $(TRAIN)$(EXT) $(TINN_SRC) $(TRAIN_SRC) $(CFLAGS) $(LDFLAGS)

predict: $(TINN_SRC) $(PREDICT_SRC)
	$(CC) -o $(PREDICT)$(EXT) $(TINN_SRC) $(PREDICT_SRC) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TRAIN)$(EXT) $(PREDICT)$(EXT)
