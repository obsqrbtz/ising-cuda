// magick convert image.pbm result.png
// add magnetic field with the mouse pointer

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <Math.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

#define BLOCKS 1024
#define THREADS 1024

#define N 1024

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

int *lattice;
int n = N;

__global__ void metropolis_step(int *spins, int reminder, int offset){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init((unsigned long long)clock() + idx, 0, 0, &state);
	int j = idx % N, i = (idx - j) / N;
	if ((i+offset) % 3 == reminder && j % 3 == reminder){
		float H = -(J) * spins[idx] * (spins[UP] + spins[DOWN] + spins[LEFT] + spins[RIGHT] + (spins[UPLEFT] + spins[UPRIGHT] + spins[DOWNLEFT] + spins[DOWNRIGHT]) / powf(sqrtf(2), 3));
		if (H > 0 || curand_uniform(&state) < expf(2 * H / TEMP)) spins[idx] *= -1;
	}
	__syncthreads();
}

int arrIdx(int i, int j){
	return i * N + j;
}

int main(void){
// Host variables
	int size_i = n * n * sizeof(int);
	lattice = new int[n * n];
	std::ofstream pbm;
	std::clock_t timer;
// Device variables
	int *lattice_d;	

	timer = clock();
	
	cudaMalloc((void**)&lattice_d, size_i);

	for (int i = 0; i < n * n; i++){
		if ((((double) rand() / (RAND_MAX))) < 0.5) lattice[i] = 1;
		else lattice[i] = -1;
	}

	cudaMemcpy(lattice_d, lattice, size_i, cudaMemcpyHostToDevice);

	for (int i = 0; i < SWEEPS; i++){
		for (int offset = 0; offset < 3; offset++){
			for (int reminder = 0; reminder < 3; reminder++) metropolis_step<<<BLOCKS, THREADS>>>(lattice_d, reminder, offset);
		}
	}
	cudaMemcpy(lattice, lattice_d, size_i, cudaMemcpyDeviceToHost);
	std::cout << std::setprecision(5) << (clock() - timer) / (double) CLOCKS_PER_SEC << "s";
    pbm.open ("output.pbm");
    pbm << "P1\n" << N << " " << N << "\n";
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (lattice[arrIdx(i, j)] == 1) pbm << 1;
            else pbm << 0;
        }
    }
    pbm.close();
	cudaFree(lattice_d);
	return 0;
}