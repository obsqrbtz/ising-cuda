// nvcc .\ising_parallel_new.cu -lgdi32 -luser32

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <windows.h>
#include <Math.h>
#include <cstdio>
#include <ctime>

#define BLOCKS 2304
#define THREADS 1024

#define N 1536

#define SWEEPS 100

// Tc = 2.269
#define TEMP 0.1
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

__global__ void metropolis_step(int *spins, int reminder, int offset){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init((unsigned long long)clock() + idx, 0, 0, &state);
	int j = idx % N, i = (idx - j) / N;
	//if ((i + j) % 2 == reminder){
	if ((i+offset) % 3 == reminder && j % 3 == reminder){
		int H = -(J) * spins[idx] * (spins[UP] + spins[DOWN] + spins[LEFT] + spins[RIGHT] + (spins[UPLEFT] + spins[UPRIGHT] + spins[DOWNLEFT] + spins[DOWNRIGHT]) / powf(sqrtf(2), 3));
		if (H > 0 || curand_uniform(&state) < expf(2 * H / TEMP)) spins[idx] *= -1;
	}
	//}
	__syncthreads();
}


int main(void){
// Host variables
	int size_i = n * n * sizeof(int);
	lattice = new int[n * n], lattice_start = new int[n * n];

// Device variables
	int *lattice_d;	
	
	cudaMalloc((void**)&lattice_d, size_i);

	for (int i = 0; i < n * n; i++){
		if ((((double) rand() / (RAND_MAX))) < 0.5) lattice[i] = 1;
		else lattice[i] = -1;
		lattice_start[i] = lattice[i];
	}
	cudaMemcpy(lattice_d, lattice, size_i, cudaMemcpyHostToDevice);

	time_t givemetime = time(NULL);
	printf("%s", ctime(&givemetime));

	for (int i = 0; i < SWEEPS; i++){
		for (int offset = 0; offset < 3; offset++){
			for (int reminder = 0; reminder < 3; reminder++) metropolis_step<<<BLOCKS, THREADS>>>(lattice_d, reminder, offset);
		}
	}
	cudaMemcpy(lattice, lattice_d, size_i, cudaMemcpyDeviceToHost);
	givemetime = time(NULL);
	printf("%s", ctime(&givemetime));
	cudaFree(lattice_d);
	return 0;
}