# This Makefile compiles the C-parts of the python program dual_utils.py
.POSIX:

all: dual_utils.so

clean:
	-rm -f dual_utils.so

CC=gcc
CFLAGS=-W -Wall -Wshadow
LIBS=-lm

dual_utils.so: dual_utils.c
	$(CC) $(CFLAGS) -fPIC -O3 -march=native -shared -o dual_utils.so dual_utils.c $(LIBS)
