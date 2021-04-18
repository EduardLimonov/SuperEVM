#include "cuda_runtime.h"
#include <thrust/device_vector.h>

#define TRIANG_COLOR 2
#define CIRCLE_COLOR 3

struct Point
{
	int x, y;

	__device__ Point(int p_x, int p_y) : x(p_x), y(p_y)
	{
	}
	__device__ Point() : x(0), y(0) {

	}
	__device__ bool operator==(const Point& other) const
	{
		return x == other.x && y == other.y;
	}
	__device__ Point operator+(const Point& other) const
	{
		return Point(x + other.x, y + other.y);
	}
	__device__ Point operator*(const int other) const
	{
		return Point(x * other, y * other);
	}
	__device__ bool is_neigh(const Point other) const
	{
		return _abs(x - other.x) <= 1 || _abs(y - other.y) <= 1;
	}

};

/*
-> counterы
-> все нити должны работать
-> coord % iter
};*/

__global__ void run(int polySize, int* polygon, Point d, int* circles, int* triangs, int nXY);
__device__ int make_cycle(int idx, int polySize, int* polygon, Point start, Point end,
	thrust::device_vector<int>& done);
__device__ thrust::device_vector<int> neighs(Point pos, Point& start, Point& stop, int polySize, int* polygon);

__device__ Point getXY();
__device__ bool contains(thrust::device_vector<int> &v, int p);
__device__ Point getByCoords(int idx, int polySize);
__device__ int getByPoint(Point p, int polySize);
__device__ void addNeigh(thrust::device_vector<int>& stack, Point pos, Point &start, Point &stop,
	thrust::device_vector<int>& done, int polySize, int* polygon);

__device__ thrust::device_vector<int>* getAwayPoint(thrust::device_vector<int>* away, Point d = {0, 0});

__device__ bool onBorder(Point t, Point& start, Point& end);
__device__ bool outBorder(Point t, Point& start, Point& end);
__device__ int _abs(int x);