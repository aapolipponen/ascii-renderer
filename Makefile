CC = gcc
CFLAGS = -Wall -g -lm

SRC = main.c
OBJ = $(SRC:.c=.o)
EXE = a.out

all: $(EXE)

$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean
