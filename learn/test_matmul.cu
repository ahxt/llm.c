#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// CUDA kernel function for matrix multiplication (M x N = M x K * K x N)
__global__ void matmul_kernel(float* out, float* A, float* B, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Determine the row of the output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Determine the column of the output matrix

    if(row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            printf("Thread (%d, %d) - A[%d, %d] * B[%d, %d] = %.0f * %.0f = %.0f\n", row, col, row, k, k, col, A[row * K + k], B[k * N + col], A[row * K + k] * B[k * N + col]);
            printf("blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, threadIdx.y = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        }
        out[row * N + col] = sum;
        printf("Thread (%d, %d) - Sum = %.0f\n", row, col, sum);
    }
}



int main() {

    int M = 4, K = 3, N = 4;  // Dimensions of matrices A (4x2), B (2x4), and C (4x4)

    float h_A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // Host array for matrix A
    float h_B[12] = {1, 3, 5, 7, 2, 4, 6, 8, 9, 10, 11, 12};  // Host array for matrix B
    float h_C[16];                            // Host array for matrix C (result)


    // Device memory pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Execution configuration
    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks((N + 1) / 2, (M + 1) / 2);

    printf("Number of blocks: (%d, %d)\n", numBlocks.x, numBlocks.y);
    printf("Number of threads per block: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);

    // Launch the kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, K, N);

    // Copy back the result to host memory
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the results (optional)
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < 12; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\n\nMatrix B:" << std::endl;
    for (int i = 0; i < 12; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << "\n\nMatrix C:" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
