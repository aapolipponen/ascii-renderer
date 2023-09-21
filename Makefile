CC = gcc
CFLAGS = -Wall -g -lm

SRC = main.c
OBJ = $(SRC:.c=.o)
EXE = a.out

MODEL_FILES = model1.obj model2.obj

all: $(EXE)

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
