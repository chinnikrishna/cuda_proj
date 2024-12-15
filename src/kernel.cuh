#ifndef KERNEL_CUH
#define KERNEL_CUH

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        c[tid] = a[tid] + b[tid];
    }
}
#endif