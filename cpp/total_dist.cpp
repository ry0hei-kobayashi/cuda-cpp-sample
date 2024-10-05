#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

struct Point {
    float x, y;
};

float calculateTotalDistance(const vector<Point>& points) {
    float totalDistance = 0.0f;
    for (size_t i = 0; i < points.size() - 1; ++i) {
        float dx = points[i + 1].x - points[i].x;
        float dy = points[i + 1].y - points[i].y;
        totalDistance += sqrt(dx * dx + dy * dy);
    }
    return totalDistance;
}

int main() {
    // sample points
    vector<Point> points(100000);
    for (size_t i = 0; i < points.size(); ++i) {
        points[i].x = static_cast<float>(i);
        points[i].y = static_cast<float>(i) / 2;
    }

    auto start = chrono::high_resolution_clock::now();

    // calc total dist
    float totalDistance = calculateTotalDistance(points);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float> duration = end - start;

    cout << "Total Distance: " << totalDistance << endl;
    cout << "C++ Execution Time: " << duration.count() << " seconds" << endl;

    return 0;
}

