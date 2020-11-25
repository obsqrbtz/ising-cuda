#define EXPORT_PBM false
#define SHOW_PROGRESSBAR false

#define BLOCKS 16384
#define THREADS 1024

#define N 16384

#define SWEEPS 100

// Tc = 2.269
#define TEMP 1.0
#define J 1

#define UP ((i - 1 + N) % N) * N + j
#define DOWN ((i + 1) % N) * N + j
#define LEFT i * N + (j - 1 + N) % N
#define RIGHT i * N + (j + 1) % N
#define UPLEFT ((i - 1 + N) % N) * N + (j - 1 + N) % N
#define UPRIGHT ((i - 1 + N) % N) * N + (j + 1) % N
#define DOWNLEFT ((i + 1) % N) * N + (j - 1 + N) % N
#define DOWNRIGHT ((i + 1) % N) * N + (j + 1) % N