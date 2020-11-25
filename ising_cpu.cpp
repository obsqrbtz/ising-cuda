// magick convert image.pbm result.png

#include <Math.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "defines.h"

#pragma warning(disable : 4996)

int *lattice;
int n = N;

int arrIdx(int i, int j){
	return i * N + j;
}

int main(void) {
	// Host variables
	float progress = 0.0;
	int size_i = n * n * sizeof(int), barWidth = 70;;
	lattice = new int[n * n]
	;
	std::ofstream pbm;
	std::clock_t timer;

	timer = clock();

	for (int i = 0; i < n * n; i++) {
		if ((((double)rand() / (RAND_MAX))) < 0.5) lattice[i] = 1;
		else lattice[i] = -1;
	}
// pow(sqrt(2), 3) = 2.828427124746f
	for (int k = 0; k < SWEEPS; k++) {
		for (int idx = 0; idx < N * N; idx++) {
			int i = idx / N, j = idx % N;
			float H = -(J) * lattice[idx] * (lattice[UP] + lattice[DOWN] + lattice[LEFT] + lattice[RIGHT] 
			+ (lattice[UPLEFT] + lattice[UPRIGHT] + lattice[DOWNLEFT] + lattice[DOWNRIGHT]) / 2.828427124746f);
			if (H > 0 || (((double)rand() / (RAND_MAX))) < expf(2 * H / TEMP)) lattice[idx] *= -1;
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

	std::cout << std::setprecision(10) << "\n\n" << (clock() - timer) / (float) CLOCKS_PER_SEC << "s\n\n";
	if (EXPORT_PBM){
		pbm.open ("output_cpu.pbm");
		pbm << "P1\n" << N << " " << N << "\n";
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++){
				if (lattice[arrIdx(i, j)] == 1) pbm << 1;
				else pbm << 0;
			}
		}
		pbm.close();
	}
	return 0;

}