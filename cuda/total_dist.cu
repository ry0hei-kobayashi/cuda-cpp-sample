#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

struct Point {
    float x, y;
};

// CUDA kernel function
__global__ void calculateDistance(Point* points, float* distances, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints - 1) {
        float dx = points[i + 1].x - points[i].x;
        float dy = points[i + 1].y - points[i].y;
        distances[i] = sqrt(dx * dx + dy * dy);
    }
}

int main() {
    const int numPoints = 100000;
    std::vector<Point> h_points(numPoints);

    // sample points
    for (int i = 0; i < numPoints; ++i) {
        h_points[i].x = static_cast<float>(i);
        h_points[i].y = static_cast<float>(i) / 2;
    }

    // mem alloc gpu
    Point* d_points;
    float* d_distances;
    cudaMalloc(&d_points, numPoints * sizeof(Point));
    cudaMalloc(&d_distances, (numPoints - 1) * sizeof(float));

    // send data to gpu
    cudaMemcpy(d_points, h_points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    // num of block and thread 
    int threadsPerBlock = 256;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    // call CUDA func
    calculateDistance<<<numBlocks, threadsPerBlock>>>(d_points, d_distances, numPoints);

    // mem copy gpu to cpu
    std::vector<float> h_distances(numPoints - 1);
    cudaMemcpy(h_distances.data(), d_distances, (numPoints - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // total dist
    float totalDistance = 0.0f;
    for (float d : h_distances) {
        totalDistance += d;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "Total Distance: " << totalDistance << std::endl;
    std::cout << "CUDA Execution Time: " << duration.count() << " seconds" << std::endl;

    // gpu mem free 
    cudaFree(d_points);
    cudaFree(d_distances);

    return 0;
}

