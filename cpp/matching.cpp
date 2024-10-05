#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cfloat>

using namespace std;

struct Point {
    float x, y;
};

// calc dist
float calculateDistance(const Point& p1, const Point& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// find the most nearest point
void matchPoints(const vector<Point>& source, const vector<Point>& target, vector<int>& matches) {
    for (size_t i = 0; i < source.size(); ++i) {
        float minDist = FLT_MAX;
        int minIndex = -1;
        for (size_t j = 0; j < target.size(); ++j) {
            float dist = calculateDistance(source[i], target[j]);
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

    // sample point 
    vector<Point> source(numPoints), target(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        source[i].x = static_cast<float>(i) / 100;
        source[i].y = static_cast<float>(i) / 200;
        target[i].x = static_cast<float>(i + 1) / 100;
        target[i].y = static_cast<float>(i + 2) / 200;
    }

    vector<int> matches(numPoints);

    auto start = chrono::high_resolution_clock::now();

    // matching
    matchPoints(source, target, matches);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float> duration = end - start;

    cout << "C++ Execution Time: " << duration.count() << " seconds" << endl;

    return 0;
}

