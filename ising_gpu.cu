// magick convert image.pbm result.png

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <Math.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "defines.h"

int *lattice;
int n = N;

__device__ float rand_d(int idx){
	curandState state;
	curand_init((unsigned long long)clock() + idx, 0, 0, &state);
	return curand_uniform(&state);
}

__global__ void init(int *spins){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (rand_d(idx) < 0.5) spins[idx] = 1;
	else spins[idx] = -1;
	__syncthreads();
}
// pow(sqrt(2), 3) = 2.828427124746f
__global__ void metropolis_step(int *spins, int reminder, int offset){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int j = idx % N, i = (idx - j) / N;
	if ((i+offset) % 3 == reminder && j % 3 == reminder){
		float H = -(J) * spins[idx] * (spins[UP] + spins[DOWN] + spins[LEFT] + spins[RIGHT] + (spins[UPLEFT] + spins[UPRIGHT] + spins[DOWNLEFT] + spins[DOWNRIGHT]) / 2.828427124746f);
		if (H > 0 || rand_d(idx) < expf(2 * H / TEMP)) spins[idx] *= -1;
	}
	__syncthreads();
}

int arrIdx(int i, int j){
	return i * N + j;
}

int main(void){
// Host variables
	float progress = 0.0;
	int size_i = n * n * sizeof(int), barWidth = 70;
	lattice = new int[n * n];
	std::ofstream pbm;
	std::clock_t timer;
// Device variables
	int *lattice_d;	

	timer = clock();
	
	cudaMalloc((void**)&lattice_d, size_i);

	std::cout << std::setprecision(5) << " \n cudaMalloc: " << (clock() - timer) / (double) CLOCKS_PER_SEC << "s\n\n";

	init<<<BLOCKS, THREADS>>>(lattice_d);
	
	for (int i = 0; i < SWEEPS; i++){
		for (int offset = 0; offset < 3; offset++){
			for (int reminder = 0; reminder < 3; reminder++) metropolis_step<<<BLOCKS, THREADS>>>(lattice_d, reminder, offset);
		}
		if (SHOW_PROGRESSBAR){
			std::cout << " [";
			int pos = barWidth * progress;
			for (int k = 0; k < barWidth; k++) {
				if (k < pos) std::cout << "=";
				else if (k == pos) std::cout << ">";
				else std::cout << " ";
			}
			std::cout << "] " << int(progress * 100.0) << " %\r";
			std::cout.flush();
			progress += 1.0f / (SWEEPS - 1);
		}
	}
	cudaMemcpy(lattice, lattice_d, size_i, cudaMemcpyDeviceToHost);
	std::cout << std::setprecision(10) << "\n\n total: "  << (clock() - timer) / (float) CLOCKS_PER_SEC << "s \n\n";
	
	if (EXPORT_PBM){
		pbm.open ("output.pbm");
		pbm << "P1\n" << N << " " << N << "\n";
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				if (lattice[arrIdx(i, j)] == 1) pbm << 1;
				else pbm << 0;
			}
		}
		pbm.close();
	}

	cudaFree(lattice_d);
	return 0;
}