#include "SimpleCounter.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void lookInside(int polySize, int* polygon, Point d, thrust::device_vector<int>* away, int* inside)
{
	// inside -- указатель на счетчик; away -- двумерный массив векторов, каждый из которых отвечает за обрабатываемый сегмент
	Point xy = getXY();
	Point start = Point(xy.x * d.x, xy.y * d.y);
	Point end = start + d;

	thrust::device_vector<int> done;

	for (int x = start.x; x < end.x; x++)
		for (int y = start.y; y < end.y; y++)
		{
			int idx = getByPoint(Point(x, y), polySize);
			if (
				(polygon[idx] == CIRCLE_COLOR || polygon[idx] == TRIANG_COLOR) && 
				!contains(done, idx)
			   )
				make_cycle(idx, polySize, polygon, away, start, end, done, inside);
		}
}

__global__ void edges(int nXY, int polySize, int* polygon, Point d, thrust::device_vector<int>* away, int* inside)
{
	// ЗАПУСКАТЬ ДЛЯ ОДНОГО БЛОКА   !!!

	// inside -- массив счетчиков
	// nXY -- количество сегментов (квадратов) по вертикали (и горизонтали)

	// выполняем редукцию
	for (int iter = 2; iter < nXY; iter *= 2)
	{
		// iter -> управляем нитью, отвечающей за квадрат iter * iter сегментов
		Point xy = getXY();
		if (xy.x == 0 && xy.y == 0 || iter % xy.x == 0 && iter % xy.y == 0) 
		{
			// это наша нить
			Point me(xy),
				right(xy.x + iter / 2, xy.y),
				bottom(xy.x, xy.y + iter / 2),
				right_bottom(xy.x + iter / 2, xy.y + iter / 2);

			thrust::device_vector<int>& mv = *getAwayPoint(away),
				& mr = *getAwayPoint(away, right),
				& mb = *getAwayPoint(away, bottom),
				& mrb = *getAwayPoint(away, right_bottom);

			mv.insert(mv.end(), mr.begin(), mr.end());
			mv.insert(mv.end(), mb.begin(), mb.end());
			mv.insert(mv.end(), mrb.begin(), mrb.end());

			mr.clear();
			mb.clear();
			mrb.clear();

			*inside += pop_neighs(mv, me, right_bottom, Point(me.x + iter, me.y + iter), d, polySize);
		}
	}

}

__device__ int pop_neighs(thrust::device_vector<int>& v, Point start, Point center, Point end, Point d,
	int polySize)
{
	int res = 0;

	for (int i = 0; i < v.size(); i++)
	{
		Point p1 = getByCoords(v[i], polySize);
		for (int j = i; j < v.size(); j++)
		{
			Point p2 = getByCoords(v[j], polySize);
			if (p1.is_neigh(p2) && !onSide(p1, p2, Point(center.x * d.x, center.y * d.y)))
			{
				res++;
				v.erase(v.begin() + i);
				v.erase(v.begin() + j);
				i--;
				j--;
			}
		}
	}
	return res;
}

__device__ bool onSide(Point p1, Point p2, Point center)
{
	// точки находятся на одной стороне
	if (
		//p1.x == p2.x && (p1.x == start.x || p1.x == end.x - 1) && (p1.y - center.y) * (p2.y - center.y) != 0 ||
		//p1.y == p2.y && (p1.y == start.y || p1.y == end.y - 1) && (p1.x - center.x) * (p2.x - center.x) != 0 ||
		(p1.y - center.y) * (p2.y - center.y) != 0 || (p1.x - center.x) * (p2.x - center.x) != 0
		)
		return true;
	else
		return false;
}

__device__ void make_cycle(int idx, int polySize, int* polygon, thrust::device_vector<int>* away, 
	Point start, Point end, thrust::device_vector<int>& done, int* inside)
{
	thrust::device_vector<int> todo;
	Point t = getByCoords(idx, polySize);
	todo.push_back(idx);
	bool get_away = false;

	while (todo.size())
	{
		int next = todo.back();
		todo.pop_back();

		done.push_back(next);
		if (onBorder(getByCoords(next, polySize), start, end))
		{
			getAwayPoint(away)->push_back(next);
			get_away = true;
		}
		addNeigh(todo, getByCoords(next, polySize), start, end, done, polySize, polygon);
	}

	if (!get_away)
		(*inside)++;
}

__device__ bool onBorder(Point t, Point& start, Point& end)
{
	return t.x == start.x || t.y == start.y || t.x == end.x || t.y == end.y;
}

__device__ bool outBorder(Point t, Point& start, Point& end)
{
	return t.x < start.x || t.y < start.y || t.x > end.x || t.y > end.y;
}

__device__ bool contains(thrust::device_vector<int> &v, int p)
{
	for (auto point : v)
		if (point == p)
			return true;
	return false;
}

__device__ void addNeigh(thrust::device_vector<int>& stack, Point pos, Point& start, Point& stop,
	thrust::device_vector<int>& done, int polySize, int* polygon)
{
	// Добавить в стэк еще не обойденных соседей
	int pos_idx = getByPoint(pos, polySize);

	for (int x = pos.x - 1; x <= pos.x + 1; x++)
		for (int y = pos.y - 1; y <= pos.y + 1; y++)
		{
			Point xy(x, y);
			if (xy == pos)
				continue;
			int idx = getByPoint(xy, polySize);
			if (polygon[idx] == polygon[pos_idx] && !contains(done, idx) && !outBorder(pos, start, stop))
				stack.push_back(idx);
		}
}

__device__ int abs(int x)
{
	return x > 0 ? x : -x;
}


// ====================================================================================================

__device__ Point getXY()
{
	return Point(threadIdx.x, blockIdx.x);
}

__device__ thrust::device_vector<int>* getAwayPoint(thrust::device_vector<int>* away, Point d)
{
	return away + (threadIdx.x + d.x) + (blockIdx.x + d.y) * gridDim.x;
}

__device__ int getByPoint(Point p, int polySize)
{
	return p.y * polySize + p.x;
}

__device__ Point getByCoords(int idx, int polySize)
{
	return Point(idx % polySize, idx / polySize);
}