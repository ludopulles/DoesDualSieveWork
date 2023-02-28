#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FT double

int num_dual_vectors, dimension, *dual_db = NULL;

static inline int cmp_FT(const void *a, const void *b)
{
	FT x = *(FT*) a;
	FT y = *(FT*) b;
	return x < y ? -1 : (x > y ? 1 : 0);
}

/*
 * Performs the Walsh-Hadamard Transform in time O(n 2^n) on `table` assumed to be of size 2^n.
*/
static inline void walsh_hadamard_transform(FT *table, const int n)
{
	FT x;

	// unroll level i=0:
	for (int j = 0; j < (1 << n); j += 2) {
		x = table[j|1];
		table[j|1] = table[j] - x;
		table[j] += x;
	}

	// unroll level i=1:
	for (int j = 0; j < (1 << n); j += 4) {
		x = table[j|2];
		table[j|2] = table[j] - x;
		table[j] += x;

		x = table[j|3];
		table[j|3] = table[j|1] - x;
		table[j|1] += x;
	}

	for (int i = 2, p2 = 1 << i; i < n; i++, p2 <<= 1) {
		for (int j = 0, nj = p2 << 1; j < (1 << n); j = nj, nj += 2*p2) {
			for (int l = j, r = j + p2; r < nj; l++, r++) {
				x = table[r];
				table[r] = table[l] - x;
				table[l] += x;
			}
		}
	}

	/* for (int i = 0; i < n; i++) {
		for (int j = 0; j < (1 << n); ) {
			// TODO: improve this code, removing this if-statement.
			if ((j >> i) & 1) { j++; k++; continue; }

			x = table[j], y = table[k];
			table[j++] = x + y;
			table[k++] = x - y;
		}
	} */
}

/*
 * Clears the database of dual vectors.
 */
void clean_up()
{
	if (dual_db != NULL) {
		free(dual_db);
	}
}

/*
 * Initializes the database of dual vectors.
 */
void init_dual_database(int _num_dual_vectors, int _dimension, int32_t *encoded)
{
	num_dual_vectors = _num_dual_vectors;
	dimension = _dimension;

	// Make a copy of the database
	clean_up();
	dual_db = (int*) calloc(num_dual_vectors * dimension, sizeof(int));
	memcpy(dual_db, encoded, num_dual_vectors * dimension * sizeof(int));
}

/* ctypes function */
int FFT_scores(int k_fft, FT threshold, FT *target, FT *FFT_table, int64_t *buckets)
{
	if (dual_db == NULL) {
		printf("ERROR: dual vector database is not initialized with init_dual_database()!");
		exit(1);
	}

	// Initialize the table with zeros.
	memset(FFT_table, (int) 0, sizeof(FT) << k_fft);

	// Initialize the database.
	for (int i = 0, k0 = 0; i < num_dual_vectors; i++) {
		int index = 0;
		FT dot_product = 0.0;
		for (int j = 0, k1 = k0; j < k_fft; j++) {
			index |= (dual_db[k1++] & 1) << j;
		}
		for (int j = 0; j < dimension; j++) {
			dot_product += target[j] * dual_db[k0++];
		}
		FFT_table[index] += cos(2 * M_PI * dot_product);
	}

	// Perform the Walsh-Hadamard Transform.
	walsh_hadamard_transform(FFT_table, k_fft);

	size_t table_size = 1U << k_fft;
	for (size_t i = 0; i < table_size; ) {
		if (FFT_table[i] <= threshold) {
            int idx = (int64_t)floor(FFT_table[i]);
            if (idx >= 0) {
                (*(buckets + idx))++;
            }
			FFT_table[i] = FFT_table[--table_size];
		} else {
			i++;
		}
	}

	// Sort the table and return the result.
	qsort(FFT_table, table_size, sizeof(FT), cmp_FT);

	return table_size;
}
