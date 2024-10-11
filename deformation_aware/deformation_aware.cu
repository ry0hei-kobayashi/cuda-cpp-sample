#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cfloat>
#include <cuda_runtime.h>

struct Point {
    float x, y, z; 
    unsigned char r, g, b;
};

// CUDA kernel for matching points and calculating total distance
__global__ void matchPointsAndCalculateDistance(Point* source, Point* target, float* distances, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        float minDist = FLT_MAX;
        for (int j = 0; j < numPoints; ++j) {
            float dx = source[i].x - target[j].x;
            float dy = source[i].y - target[j].y;
            float dz = source[i].z - target[j].z;
            float dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < minDist) {
                minDist = dist;
            }
        }
        distances[i] = minDist;
    }
}

// Check for CUDA errors
void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to load point cloud from a text file
std::vector<Point> loadPointCloud(const std::string& filename) {
    std::vector<Point> points;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Point point;
        float r, g, b;  // Read color as float, then convert to unsigned char
        if (!(iss >> point.x >> point.y >> point.z >> r >> g >> b)) {
            std::cerr << "Error: Invalid data format in " << filename << std::endl;
            break;
        }
        point.r = static_cast<unsigned char>(r);
        point.g = static_cast<unsigned char>(g);
        point.b = static_cast<unsigned char>(b);
        points.push_back(point);
    }
    file.close();
    return points;
}

int main() {
    // Load point clouds from files in the samples directory
    std::vector<Point> h_source = loadPointCloud("samples/milk_carton_05.txt");
    std::vector<Point> h_target = loadPointCloud("samples/milk_carton_15.txt");

    int numPointsSource = h_source.size();
    int numPointsTarget = h_target.size();

    if (numPointsSource != numPointsTarget) {
        std::cerr << "Warning: Point clouds have different number of points. Adjusting processing to smaller set." << std::endl;
        numPointsSource = std::min(numPointsSource, numPointsTarget);  // Use the smaller set of points
    }

    // Allocate memory on the GPU
    Point* d_source;
    Point* d_target;
    float* d_distances;

    checkCudaError(cudaMalloc(&d_source, numPointsSource * sizeof(Point)), "Failed to allocate device memory for source");
    checkCudaError(cudaMalloc(&d_target, numPointsSource * sizeof(Point)), "Failed to allocate device memory for target");
    checkCudaError(cudaMalloc(&d_distances, numPointsSource * sizeof(float)), "Failed to allocate device memory for distances");

    // Copy data to GPU
    checkCudaError(cudaMemcpy(d_source, h_source.data(), numPointsSource * sizeof(Point), cudaMemcpyHostToDevice), "Failed to copy source data to GPU");
    checkCudaError(cudaMemcpy(d_target, h_target.data(), numPointsSource * sizeof(Point), cudaMemcpyHostToDevice), "Failed to copy target data to GPU");

    // Define thread and block sizes
    int threadsPerBlock = 256;
    int numBlocks = (numPointsSource + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch the CUDA kernel
    matchPointsAndCalculateDistance<<<numBlocks, threadsPerBlock>>>(d_source, d_target, d_distances, numPointsSource);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Wait for the kernel to finish
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization failed");

    // Copy the distances back to the CPU
    std::vector<float> h_distances(numPointsSource);
    checkCudaError(cudaMemcpy(h_distances.data(), d_distances, numPointsSource * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy distances data to CPU");

    // Calculate total distance
    float totalDistance = 0.0f;
    for (int i = 0; i < numPointsSource; ++i) {
        totalDistance += h_distances[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "CUDA Execution Time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Total Distance: " << totalDistance << std::endl;

    // Free GPU memory
    checkCudaError(cudaFree(d_source), "Failed to free source memory");
    checkCudaError(cudaFree(d_target), "Failed to free target memory");
    checkCudaError(cudaFree(d_distances), "Failed to free distances memory");

    return 0;
}

