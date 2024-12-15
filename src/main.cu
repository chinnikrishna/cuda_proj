#include <stdio.h>
#include <nvToolsExt.h>
#include "kernel.cuh"


int main() {
    const int N = 1024;
    float *ha, *hb, *hc;
    float *da, *db, *dc;

    // Allocate Host memory
    // malloc returns a pointer to start of allocated space
    // Cast the pointer to be of type ha
    ha = (float*) malloc(N * sizeof(float));
    hb = (float*) malloc(N * sizeof(float));
    hc = (float*) malloc(N * sizeof(float));

    // Initialize these arrays
    for (int i=0; i<N; ++i){
        ha[i] = i;
        hb[i] = 2 * i;
    }

    // Allocate Device memory
    // Why are we giving address of a pointer to a pointer?
    cudaMalloc(&da, N * sizeof(float));
    cudaMalloc(&db, N * sizeof(float));
    cudaMalloc(&dc, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(da, db, dc, N);

    // Move result back to host
    cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    for(int i=0; i < N; ++i){
        if(hc[i] != ha[i] + hb[i]){
            printf("Verification failed at element %d\n", i);
            break;
        }
    }
    printf("Done");


    // Cleanup
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);

    return 0;
}