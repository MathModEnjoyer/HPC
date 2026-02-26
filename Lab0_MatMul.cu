#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cstdlib> 
#include <ctime>
#include <iomanip>
#include <string>

using namespace std;

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void matmul_gpu(const float* A, const float* B, float* C,
    int M, int N, int K) {

    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N && idy < M) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[idy * K + k] * B[k * N + idx];
        }
        C[idy * N + idx] = sum;
    }
}

void generate_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (rand() % 2001 - 1000) / 100.0f;  // от -10.00 до 10.00
    }
}

bool check_result(const float* C_cpu, const float* C_gpu, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(C_cpu[i] - C_gpu[i]) > 1e-3f) {
            return false;
        }
    }
    return true;
}

int main() {
    int sizes[] = { 100, 500, 1000, 2000 };
    int iterations = 12;
    srand(time(nullptr));

    cout << left
        << setw(10) << "Size"
        << setw(10) << "CPU(ms)"
        << setw(10) << "GPU(ms)"
        << setw(9) << "Speedup"
        << "Correct" << endl;

    for (int size : sizes) {
        int M = size, N = size, K = size;
        int n = M * N;

        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C_cpu = new float[n];
        float* C_gpu = new float[n];

        float* d_A, * d_B, * d_C;
        size_t size_A = M * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = M * N * sizeof(float);

        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        double cpu_total = 0, gpu_total = 0;
        bool ok = true;

        cudaEvent_t cuda_start, cuda_stop;
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);

        for (int iter = 0; iter < iterations; ++iter) {
            generate_matrix(A, M, K);
            generate_matrix(B, K, N);

            auto start = chrono::high_resolution_clock::now();
            matmul_cpu(A, B, C_cpu, M, N, K);
            auto end = chrono::high_resolution_clock::now();
            cpu_total += chrono::duration<double, milli>(end - start).count();

            cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
            
            cudaEventRecord(cuda_start);
            matmul_gpu << <grid, block >> > (d_A, d_B, d_C, M, N, K);
            cudaEventRecord(cuda_stop);
            cudaDeviceSynchronize();

            float gpu_time;
            cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);
            gpu_total += gpu_time;

            cudaMemcpy(C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

            if (!check_result(C_cpu, C_gpu, n)) ok = false;
        }

        double avg_cpu = cpu_total / iterations;
        double avg_gpu = gpu_total / iterations;
        double speedup = avg_cpu / avg_gpu;

        cout << fixed << setprecision(2)
            << left << setw(10) << (to_string(size) + "x" + to_string(size))
            << setw(10) << avg_cpu
            << setw(10) << avg_gpu
            << setw(9) << speedup
            << "  " << (ok ? "OK" : "ERR") << endl;

        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        delete[] A;
        delete[] B;
        delete[] C_cpu;
        delete[] C_gpu;
    }
    return 0;
}
