#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>



__global__ void addMatrixKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] + B[index];
        printf("Thread (%d, %d) - A[%d, %d] + B[%d, %d] = %.1f + %.1f = %.1f\n", row, col, row, col, row, col, (float)A[index], (float)B[index], (float)C[index]);
        printf("blockIdx (%d, %d), blockDim (%d, %d), threadIdx (%d, %d)\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
    }
}


int main() {
    // Define matrix dimensions (2x2 matrices)
    int numRows = 4;  
    int numCols = 4;  
    int numElements = numRows * numCols;
    size_t size = numElements * sizeof(__nv_bfloat16);

    // Define and initialize host matrices
    __nv_bfloat16 h_A[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    __nv_bfloat16 h_B[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    __nv_bfloat16 h_C[16];  // Result matrix C

    // Allocate device arrays
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set execution configuration
    dim3 threadsPerBlock(2, 2);  // Block dimensions: 2x2, perfectly fits our matrix
    dim3 numBlocks(2, 2);        // Grid dimensions: 1x1, only one block needed

    // Launch kernel
    addMatrixKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, numRows, numCols);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "Matrix C:" << std::endl;
    for (int i = 0; i < numElements; ++i) {
        std::cout << (float)h_C[i] << " ";
        if ((i + 1) % numCols == 0)
            std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
