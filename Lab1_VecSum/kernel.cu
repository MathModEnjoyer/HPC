#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;

void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void generate_vector(float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = (rand() % 2001 - 1000) / 100.0f;
    }
}

int main() {
    int sizes[] = { 1000, 50000, 500000, 10000000 };
    int iterations = 12;

    srand(time(nullptr));
   
    cout << left
        << setw(12) << "Size"
        << setw(16) << "CPU (ms)"
        << setw(16) << "GPU (ms)"
        << "Speedup" << endl;

    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    

    for (int size : sizes) {
        float* a = new float[size];
        float* b = new float[size];
        float* c_cpu = new float[size];
        float* c_gpu = new float[size];

        double cpu_total = 0, kernel_total = 0;

        for (int iter = 0; iter < iterations; ++iter) {

            float* d_a, * d_b, * d_c;
            float kernel_time;

            generate_vector(a, size);
            generate_vector(b, size);

            auto start = chrono::high_resolution_clock::now();
            vector_add_cpu(a, b, c_cpu, size);
            auto end = chrono::high_resolution_clock::now();
            cpu_total += chrono::duration<double, milli>(end - start).count();

            size_t bytes = size * sizeof(float);

            cudaMalloc(&d_a, bytes);
            cudaMalloc(&d_b, bytes);
            cudaMalloc(&d_c, bytes);

            cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (size + threads - 1) / threads;

            cudaEventRecord(cuda_start);
            vector_add_gpu << <blocks, threads >> > (d_a, d_b, d_c, size);
            cudaEventRecord(cuda_stop);
            cudaDeviceSynchronize();

            cudaEventElapsedTime(&kernel_time, cuda_start, cuda_stop);
            kernel_total += kernel_time;

            cudaMemcpy(c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }

        double avg_cpu = cpu_total / iterations;
        double avg_gpu = kernel_total / iterations;
        double speedup = avg_cpu / avg_gpu;

        cout << left << setw(12) << size
            << setw(16) << fixed << setprecision(3) << avg_cpu
            << setw(16) << avg_gpu
            << speedup << endl;

        delete[] a;
        delete[] b;
        delete[] c_cpu;
        delete[] c_gpu;
    }
    
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    return 0;
}
