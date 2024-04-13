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
        }
        out[row * N + col] = sum;
    }
}

// Function to fill the matrix with random floats
void random_fill(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 100.0;  // Random floats between 0 and 100
    }
}

int main() {
    srand(time(0));  // Seed for random number generation

    // Matrix dimensions
    int M = 512;  // Number of rows in A and C
    int K = 256;  // Number of columns in A and rows in B
    int N = 1024; // Number of columns in B and C

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    // Initialize matrices A and B with random values
    random_fill(h_A, M * K);
    random_fill(h_B, K * N);

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
    dim3 threadsPerBlock(16, 16);  // 16x16 thread block size
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);  // Ensure there are enough blocks

    // Launch the kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, K, N);

    // Copy back the result to host memory
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the results (optional)
    std::cout << "Matrix A (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\n\nMatrix B (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << "\n\nMatrix C (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
