#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cfloat>

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

int main() {
    const int numPoints = 8000;

    // sample points
    std::vector<Point> h_source(numPoints), h_target(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        h_source[i].x = static_cast<float>(i) / 100;
        h_source[i].y = static_cast<float>(i) / 200;
        h_target[i].x = static_cast<float>(i + 1) / 100;
        h_target[i].y = static_cast<float>(i + 2) / 200;
    }

    // mem copy to gpu 
    Point* d_source;
    Point* d_target;
    int* d_matches;
    cudaMalloc(&d_source, numPoints * sizeof(Point));
    cudaMalloc(&d_target, numPoints * sizeof(Point));
    cudaMalloc(&d_matches, numPoints * sizeof(int));

    cudaMemcpy(d_source, h_source.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    // num of thread and num of block
    int threadsPerBlock = 256;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    // call cuda kernel 
    matchPoints<<<numBlocks, threadsPerBlock>>>(d_source, d_target, d_matches, numPoints);

    // mem copy gpu to cpu
    std::vector<int> h_matches(numPoints);
    cudaMemcpy(h_matches.data(), d_matches, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "CUDA Execution Time: " << duration.count() << " seconds" << std::endl;

    // memory free 
    cudaFree(d_source);
    cudaFree(d_target);
    cudaFree(d_matches);

    return 0;
}

