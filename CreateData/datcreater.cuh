/*
ОПИСАНИЕ
Данный алгоритм реализует генерацию случайного полигона для задания 3.2. Полигон имеет заданный размер 
и представлен в памяти как одномерный массив. На полигоне размещены контуры до nTriang треугольников и nCircles
окружностей. 
*/
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <curand.h>

/*

|^| | |
 _ _ _ 
| |*|1|
 _ _ _
|4|3|2|

(top, bottom)

// pop_intersects -- [i]
// generate figures -- не удалять прямоугольники

*/

class curandState;

#define EXPECTED_VAL 100
#define DISPERSION 10
#define MIN_RECT_SIDE 70
#define MAX_CONFLICT 10000
#define TRING_PERIM 20000

#define TRIANG_COLOR 2
#define CIRCLE_COLOR 3

#define ACK_ANGLED // генерировать только остроугольные треугольники

struct Point
{
	int x, y;

	__device__ Point(int p_x, int p_y): x(p_x), y(p_y)
	{
	}
	__device__ Point() : x(0), y(0) {

	}
};

struct Rect
{
	Point lt, rb;

	__device__ Rect(Point p_lt, Point p_rb) : lt(p_lt), rb(p_rb) {

	}

	__device__ bool intersects(Rect& other);
};

struct Triangle
{
	Point p1, p2, p3;

	__device__ Triangle(Point pp1, Point pp2, Point pp3): p1(pp1), p2(pp2), p3(pp3)
	{

	}
};

struct Circle
{
	Point center;
	int r;

	__device__ Circle(Point c, int rad) : center(c), r(rad) {

	}
};

template <typename T>
struct Pair
{
	T first, second;
};

__global__ void create_objects(Point d, int nTriang, int nCircl, Pair<thrust::device_vector<Rect>>* conflicts, 
	int polySize, int* polygon);

__device__ void pop_intersecting(thrust::device_vector<Rect> &v);
__device__ void resolveConflicts(Pair<thrust::device_vector<Rect>>* conf); // !!!

__device__ float randUniform(float min, float max, curandState* state);
__device__ float randNorm(float ev, float disp, curandState* state);

__device__ void generateTriangles(thrust::device_vector<Triangle>& triangles, int size, thrust::device_vector<Rect>& rects, curandState* state);
__device__ void generateCircles(thrust::device_vector<Circle>& circles, int size, thrust::device_vector<Rect>& rects);

__device__ float Perimetr2(Point p1, Point p2, Point p3);
__device__ void generatePoint(Point& p, Rect& r, curandState* state);
__device__ bool onBorder(Rect& r, Point lt, Point rb);
__device__ bool onPolygonEdge(Rect& r, int polySize);

__device__ bool atLeft(Rect r, Point lt, Point rb);
__device__ bool atRight(Rect r, Point lt, Point rb);

__device__ void drawTriangles(thrust::device_vector<Triangle>& triangles, int polySize, int* polygon);
__device__ void drawCircles(thrust::device_vector<Circle>& circles, int polySize, int* polygon);
__device__ void drawPixel(int x, int y, int polySize, int* polygon, int color);
__device__ void drawLine(Point p1, Point p2, int polySize, int* polygon);

__device__ bool iSackAngled(Point& p1, Point& p2, Point& p3);

__device__ void syncAll();
