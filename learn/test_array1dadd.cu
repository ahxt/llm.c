#include <iostream>
#include <cuda_runtime.h>


__global__ void addArrayKernel(float* A, float* B, float* C, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
        printf("Thread %d - A[%d] + B[%d] = %.0f + %.0f = %.0f\n", idx, idx, idx, A[idx], B[idx], C[idx]);
        printf("blockIdx %d, blockDim %d, threadIdx %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    }
}


int main() {
    // Define array dimensions
    int numElements = 16;  
    size_t size = numElements * sizeof(float);

    // Define and initialize host arrays
    float h_A[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_C[16];  // Result array C

    // Allocate device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set execution configuration
    int threadsPerBlock = 4;  // Block dimensions: 4 threads per block
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;  // Calculate needed number of blocks

    // Launch kernel
    addArrayKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "Array C:" << std::endl;
    for (int i = 0; i < numElements; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


