#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cfloat>
#include <cuda_runtime.h>

// Struct for points
struct Point {
    float x, y;
};

// CUDA kernel
__global__ void matchPoints(Point* source, Point* target, int* matches, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        float minDist = FLT_MAX;
        int minIndex = -1;
        for (int j = 0; j < numPoints; ++j) {
            float dx = source[i].x - target[j].x;
            float dy = source[i].y - target[j].y;
            float dist = sqrt(dx * dx + dy * dy);
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }
        matches[i] = minIndex;
    }
}

// Check for CUDA errors
void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int numPoints = 8000;

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis1(-100.0, 100.0); // Random values for source points
    std::uniform_real_distribution<> dis2(-200.0, 200.0); // Random values for target points

    // Sample points with random values
    std::vector<Point> h_source(numPoints), h_target(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        h_source[i].x = dis1(gen);
        h_source[i].y = dis1(gen);
        h_target[i].x = dis2(gen);
        h_target[i].y = dis2(gen);
    }

    // Allocate memory on the GPU
    Point* d_source;
    Point* d_target;
    int* d_matches;

    checkCudaError(cudaMalloc(&d_source, numPoints * sizeof(Point)), "Failed to allocate device memory for source");
    checkCudaError(cudaMalloc(&d_target, numPoints * sizeof(Point)), "Failed to allocate device memory for target");
    checkCudaError(cudaMalloc(&d_matches, numPoints * sizeof(int)), "Failed to allocate device memory for matches");

    // Copy data to GPU
    checkCudaError(cudaMemcpy(d_source, h_source.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice), "Failed to copy source data to GPU");
    checkCudaError(cudaMemcpy(d_target, h_target.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice), "Failed to copy target data to GPU");

    // Define thread and block sizes
    int threadsPerBlock = 256;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch the CUDA kernel
    matchPoints<<<numBlocks, threadsPerBlock>>>(d_source, d_target, d_matches, numPoints);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Wait for the kernel to finish
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization failed");

    // Copy the results back to the CPU
    std::vector<int> h_matches(numPoints);
    checkCudaError(cudaMemcpy(h_matches.data(), d_matches, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy matches data to CPU");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "CUDA Execution Time: " << duration.count() << " seconds" << std::endl;

    // Free GPU memory
    checkCudaError(cudaFree(d_source), "Failed to free source memory");
    checkCudaError(cudaFree(d_target), "Failed to free target memory");
    checkCudaError(cudaFree(d_matches), "Failed to free matches memory");

    return 0;
}

