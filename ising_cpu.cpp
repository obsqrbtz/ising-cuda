// magick convert image.pbm result.png

#include <Math.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>

#pragma warning(disable : 4996)

#define N 768

#define SWEEPS 100

// Tc = 2.269
#define TEMP 1
#define J 1

#define UP ((i - 1 + N) % N) * N + j
#define DOWN ((i + 1) % N) * N + j
#define LEFT i * N + (j - 1 + N) % N
#define RIGHT i * N + (j + 1) % N
#define UPLEFT ((i - 1 + N) % N) * N + (j - 1 + N) % N
#define UPRIGHT ((i - 1 + N) % N) * N + (j + 1) % N
#define DOWNLEFT ((i + 1) % N) * N + (j - 1 + N) % N
#define DOWNRIGHT ((i + 1) % N) * N + (j + 1) % N

int *lattice, *lattice_start;
int n = N;

int arrIdx(int i, int j){
	return i * N + j;
}

int main(void) {
	// Host variables
	int size_i = n * n * sizeof(int);
	lattice = new int[n * n], lattice_start = new int[n * n];
	std::ofstream pbm;
	std::clock_t timer;

	for (int i = 0; i < n * n; i++) {
		if ((((double)rand() / (RAND_MAX))) < 0.5) lattice[i] = 1;
		else lattice[i] = -1;
		lattice_start[i] = lattice[i];
	}

	timer = clock();

	for (int k = 0; k < SWEEPS; k++) {
		for (int idx = 0; idx < N * N; idx++) {
			int i = idx / N, j = idx % N, H = -(J) * lattice[idx] * (lattice[UP] + lattice[DOWN] + lattice[LEFT] + lattice[RIGHT] 
			+ (lattice[UPLEFT] + lattice[UPRIGHT] + lattice[DOWNLEFT] + lattice[DOWNRIGHT]) / powf(sqrtf(2), 3));
			if (H > 0 || (((double)rand() / (RAND_MAX))) < expf(2 * H / TEMP)) lattice[idx] *= -1;
		}
	}

	std::cout << std::setprecision(5) << (clock() - timer) / (double) CLOCKS_PER_SEC << " sec";
	
    /*pbm.open ("output_cpu.pbm");
    pbm << "P1\n" << N << " " << N << "\n";
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (lattice[arrIdx(i, j)] == 1) pbm << 1;
            else pbm << 0;
        }
    }
    pbm.close();*/
	return 0;

}