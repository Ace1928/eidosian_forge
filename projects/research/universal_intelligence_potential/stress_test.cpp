// Save this as stress_test.cpp
#include <iostream>
#include <hip/hip_runtime.h>

__global__ void simple_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {
    const int n = 1 << 20;
    float* data;
    hipMalloc(&data, n * sizeof(float));
    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    while (true) {
        simple_kernel<<<numBlocks, threadsPerBlock>>>(data, n);
        hipDeviceSynchronize();
    }

    hipFree(data);
    return 0;
}
