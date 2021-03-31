#include "datcreater.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include "curand_kernel.h"

template<typename T1, typename T2>
constexpr auto imin(T1 a, T2  b) { return a < b ? a: b; }

bool Rect::intersects(Rect& other)
{
	if (rb.x + 1 < other.lt.x || lt.x - 1 > other.rb.x || rb.y - 1 > other.lt.y && lt.y + 1 < other.rb.y)
		// прямоугольники лежат на расстоянии друг от друга
		return false;
	return true;
}

__global__ void create_rects(Point d, int nTriang, int nCircl, Pair<thrust::device_vector<Rect>>* conflicts,
	int polySize, int* polygon, thrust::device_vector<Rect>* rects)
{
	curandState* state;
	curand_init(unsigned(time(NULL)), threadIdx.x, 0, state);
	
	generateGarbage(d, polySize, polygon, state);

	Point start(threadIdx.x * d.x, blockIdx.x * d.y); // левый верхний угол квадрата, обрабатываемого данной нитью
	Point end = Point(start.x + d.x, start.y + d.y);


	for (int i = 0; i < nTriang + nCircl; i++)
	{
		Point lt, rb;
		do
		{
			lt = { randUniform(start.x + 1, start.x + d.x, state), randUniform(start.y + 1, start.y + d.y, state) };
			rb = { abs(randNorm(EXPECTED_VAL, DISPERSION, state)) + lt.x, abs(randNorm(EXPECTED_VAL, DISPERSION, state)) + lt.y };
		} while (rb.x - lt.x > MIN_RECT_SIDE || lt.y - rb.y > MIN_RECT_SIDE);
		
		Rect nr(lt, rb);
		if (onPolygonEdge(nr, polySize))
			rects->push_back(nr);
	}
	// создали случайные прямоугольники (возможно, пересекающиеся)
	pop_intersecting(*rects); // решаем внутренние проблемы

	for (auto nr: *rects)
	{
		bool second = atRight(nr, start, end);
		if (second)
			// прямоугольник может конфликтовать с нижней дугой
			conflicts[blockIdx.x * gridDim.x + threadIdx.x].second.push_back(nr);
		else
			// прямоугольник может конфликтовать с верхней дугой
			conflicts[blockIdx.x * gridDim.x + threadIdx.x].first.push_back(nr);
	}

}

__global__ void create_objects(int nTriang, int nCircl, Pair<thrust::device_vector<Rect>>* conflicts,
	int polySize, int* polygon, thrust::device_vector<Rect>* rects, int* realTriang, int* realCirc)
{
	curandState* state;
	curand_init(unsigned(time(NULL)), threadIdx.x, 0, state);

	// resolveConflicts(conflicts); // разрешаем коллизии на границе

	rects->clear();
	rects->insert(rects->end(),
		conflicts[blockIdx.x * gridDim.x + threadIdx.x].first.begin(),
		conflicts[blockIdx.x * gridDim.x + threadIdx.x].first.end());
	rects->insert(rects->end(),
		conflicts[blockIdx.x * gridDim.x + threadIdx.x].second.begin(),
		conflicts[blockIdx.x * gridDim.x + threadIdx.x].second.end());
	// теперь все прямоугольники лежат в rects
	conflicts[blockIdx.x * gridDim.x + threadIdx.x].first.clear();
	conflicts[blockIdx.x * gridDim.x + threadIdx.x].second.clear();

	size_t old = nTriang + nCircl;
	float df = rects->size() / old;
	// теперь прямоугольники не пересекаются

	thrust::device_vector<Triangle> triangles;

	thrust::device_vector<Circle> circles;

	generateTriangles(triangles, nTriang * df, *rects, state);
	generateCircles(circles, nCircl * df, *rects);
	// векторы заполнены случайными треугольниками и кругами

	drawTriangles(triangles, polySize, polygon);
	drawCircles(circles, polySize, polygon);

	*realCirc = circles.size();
	*realTriang = triangles.size();
}

__device__ void pop_intersecting(thrust::device_vector<Rect>& v)
{	// удаляет из вектора все прямоугольники, которые пересекаются с предыдущими
	for (int i = 0; i < v.size(); i++)
		for (int j = i + 1; j < v.size(); j++)
		{
			Rect r = v[i];
			Rect r_n = v[j];
			if (r_n.intersects(r))
			{
				v.erase(v.begin() + j);
				j--;
			}
		}
}

__device__ void resolveConflicts(thrust::device_vector<Rect>& v1, thrust::device_vector<Rect>& v2)
// удаляет из v2 все прямоугольники, пересекающиеся с прямоугольниками из v1
{
	for (auto it = v1.begin(); it != v1.end(); it++)
	{
		for (auto it2 = v2.begin(); it2 != v2.end(); )
		{
			Rect r_n = *it2;
			Rect rr = *it;
			if (rr.intersects(r_n))
				it2 = v2.erase(it2);
			else
				it2++;
		}
	}
}

__device__ void resolveConflicts(Pair<thrust::device_vector<Rect>>* conf)
// решить все пограничные конфликты прямоугольников;
// first -- векторы на границе слева-сверху; second -- справа-снизу
// синхронизация удалена -- НЕ ИСПОЛЬЗОВАТЬ ЭТУ МЕРЗОСТЬ БЕЗ ДОБАВЛЕНИЯ СИНХРОНИЗАЦИИ!!!
{
	auto v1 = conf[blockIdx.x * gridDim.x + threadIdx.x].second;

	for (int i = 0; i < 4; i++)
	{
		int j;
		switch (i)
		{
		case 0: {
			// сосед справа
			j = blockIdx.x * gridDim.x + threadIdx.x + 1;
			break; 
		}
		case 1: {
			// сосед справа-снизу
			j = (blockIdx.x + 1) * gridDim.x + threadIdx.x + 1;
			break; 
		}
		case 2: {
			// сосед снизу
			j = (blockIdx.x + 1) * gridDim.x + threadIdx.x;
			break; 
		}
		case 3: {
			// сосед слева-снизу
			j = (blockIdx.x + 1) * gridDim.x + threadIdx.x - 1;
			break; 
		}
		}

		if (j < MAX_CONFLICT)
		{
			thrust::device_vector<Rect>& v2 = conf[j].first;
			resolveConflicts(v2, v1); 
		}

		// syncAll();
	}
}

__global__ void resolveConflicts(Pair<thrust::device_vector<Rect>>* conf, int nSide)
{
	auto v1 = conf[blockIdx.x * gridDim.x + threadIdx.x].second;
	int j;
	switch (nSide)
	{
	case 0: {
		// сосед справа
		j = blockIdx.x * gridDim.x + threadIdx.x + 1;
		break;
	}
	case 1: {
		// сосед справа-снизу
		j = (blockIdx.x + 1) * gridDim.x + threadIdx.x + 1;
		break;
	}
	case 2: {
		// сосед снизу
		j = (blockIdx.x + 1) * gridDim.x + threadIdx.x;
		break;
	}
	case 3: {
		// сосед слева-снизу
		j = (blockIdx.x + 1) * gridDim.x + threadIdx.x - 1;
		break;
	}
	}

	if (j < MAX_CONFLICT)
	{
		thrust::device_vector<Rect>& v2 = conf[j].first;
		resolveConflicts(v2, v1);
	}
}

__device__ void generateTriangles(thrust::device_vector<Triangle>& triangles, int size, thrust::device_vector<Rect>& rects, curandState* state)
{
	for (int cnt = 0; cnt < size; cnt++)
	{
		Rect rect = rects.back();
		rects.pop_back();
		Point p1, p2, p3;

		do
		{
			generatePoint(p1, rect, state);
			generatePoint(p2, rect, state);
			generatePoint(p3, rect, state);
		} while (Perimetr2(p1, p2, p3) < TRING_PERIM
#ifdef ACK_ANGLED
			&& iSackAngled(p1, p2, p3)
#endif
			);

		triangles.push_back(Triangle(p1, p2, p3));
	}
}

__device__ void generateCircles(thrust::device_vector<Circle>& circles, int size, thrust::device_vector<Rect>& rects)
{
	for (int cnt = 0; cnt < size; cnt++)
	{
		Rect rect = rects.back();
		rects.pop_back();
		int r = imin((rect.rb.x - rect.lt.x) / 2, (rect.lt.y - rect.rb.y) / 2);
		Point center(r + rect.lt.x, r + rect.rb.y);
		circles.push_back(Circle(center, r));
	}
	
}

__device__ float Perimetr2(Point p1, Point p2, Point p3)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + 
		(p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y) +
		(p3.x - p2.x) * (p3.x - p2.x) + (p3.y - p2.y) * (p3.y - p2.y);
}

__device__ void generatePoint(Point& p, Rect& rect, curandState* state)
{
	float r = randUniform(0, 1, state);
	int side = randUniform(0, 3.998, state);
	if (side == 0 || side == 2)
	{
		p.y = rect.rb.y + (rect.lt.y - rect.rb.y) * r;
		p.x = (side == 0) ? rect.lt.x : rect.rb.x;
	}
	else
	{
		p.x = rect.lt.x + (rect.rb.x - rect.lt.x) * r;
		p.y = (side == 1) ? rect.lt.y : rect.rb.y;
	}
}

__device__ bool onBorder(Rect& r, Point lt, Point rb)
// прямоугольник лежит недалеко от границы
{
	if (r.lt.x <= lt.x || r.lt.y >= lt.y || r.rb.x >= rb.x || r.rb.y <= rb.y)
		return true;
	return false;
}

__device__ bool onPolygonEdge(Rect& r, int polySize)
{
	if (r.lt.x <= 0 || r.lt.y >= polySize || r.rb.x >= polySize || r.rb.y <= 0)
		return true;
	return false;
}

__device__ bool atLeft(Rect r, Point lt, Point rb)
{
	return r.lt.x <= lt.x && r.lt.y > rb.y // прямоугольник слева (не лежит целиком слева-снизу)
		|| r.lt.y >= lt.y; // или сверху
}

__device__ bool atRight(Rect r, Point lt, Point rb)
{
	return r.rb.x >= rb.x && r.rb.y < lt.y || // прямоугольник справа (не лежит целиком справа-сверху)
		r.rb.y >= rb.y; // или снизу
}

/*__device__ void syncAll()
{
	using namespace cooperative_groups;
	grid_group grid = this_grid();
	grid.sync(); // синхронизация всех потоков всех блоков
}*/

__device__ bool notHyp(Point& p1, Point& p2, Point& p3)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
		(p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y) >
		(p3.x - p2.x) * (p3.x - p2.x) + (p3.y - p2.y) * (p3.y - p2.y);
}

__device__ bool iSackAngled(Point& p1, Point& p2, Point& p3)
{
	return notHyp(p1, p2, p3) && notHyp(p2, p3, p1) && notHyp(p3, p1, p2);
}

__device__ float randUniform(float min, float max, curandState* state)
{
	if (max < min)
		throw "MAX < MIN";
	//curandState localState = **state;
	float r = min + curand_uniform(state) * (max - min);
	//**state = localState;
	return r;
}

__device__ float randNorm(float ev, float disp, curandState* state)
{
	float r = ev + disp * curand_normal(state);
	return r;
}

__device__ void drawTriangles(thrust::device_vector<Triangle>& triangles, int polySize, int* polygon)
{
	for (Triangle t : triangles)
	{
		drawLine(t.p1, t.p2, polySize, polygon);
		drawLine(t.p3, t.p2, polySize, polygon);
		drawLine(t.p1, t.p3, polySize, polygon);
	}
}

__device__ void generateGarbage(Point d, int polySize, int* polygon, curandState* state)
{
	for (int i = 0; i <= d.y; i++)
		for (int j = 0; j <= d.x; j++)
		{
			int r;
			do
			{
				r = randUniform(0, _UI8_MAX, state);
			} while (r == TRIANG_COLOR || r == CIRCLE_COLOR);

			polygon[j + i * polySize] = r;
		}
}

__device__ void drawCircles(thrust::device_vector<Circle>& circles, int polySize, int* polygon)
{
	// алгоритм Брезенхема для рисования окружностей
	// Возможно, это еще хуже, чем рисование прямой
	for (Circle c : circles)
	{
		
		int x = 0;
		int y = c.r;
		int delta = 1 - 2 * c.r;
		int error = 0;
		while (y >= 0)
		{
			drawPixel(c.center.x + x, c.center.y + y, polySize, polygon, CIRCLE_COLOR);
			drawPixel(c.center.x + x, c.center.y - y, polySize, polygon, CIRCLE_COLOR);
			drawPixel(c.center.x + x, c.center.y + y, polySize, polygon, CIRCLE_COLOR);
			drawPixel(c.center.x + x, c.center.y + y, polySize, polygon, CIRCLE_COLOR);

			error = 2 * (delta + y) - 1;

			if ((delta < 0) && (error <= 0))
			{
				delta += 2 * ++x + 1;
				continue;
			}
			if ((delta > 0) && (error > 0))
			{
				delta -= 2 * --y + 1;
				continue;
				delta += 2 * (++x - --y);
			}
		}
	}
}

__device__ int abs(int x)
{
	return x > 0 ? x : -x;
}

__device__ void drawPixel(int x, int y, int polySize, int* polygon, int color)
{
	polygon[x + y * polySize] = color;
}

__device__ void drawLine(Point p1, Point p2, int polySize, int* polygon)
{
	// Алгоритм Брезенхема
	// это ужасно, но что поделаешь?..
	int dx = p2.x - p1.x, dy = p2.y - p1.y, d, x, y, d1, d2;

	if (((abs(dx) > abs(dy)) && (p2.x < p1.x)) || ((abs(dx) <= abs(dy)) && (p2.y < p1.y)))
	{
		x = p1.x;
		p1.x = p2.x;
		p2.x = x;
		y = p1.y;
		p1.y = p2.y;
		p2.y = y;

		dx = p2.x - p1.x;
		dy = p2.y - p1.y;
	}

	drawPixel(p1.x, p1.y, polySize, polygon, TRIANG_COLOR);
	int stp = 1;

	if (abs(dx) > abs(dy))
	{
		if (dy < 0)
		{
			stp = -1;
			dy = -dy;
		}

		d = dy * 2 - dx;
		d1 = dy * 2;
		d2 = (dy - dx) * 2;
		y = p1.y;

		for (x = p1.x + 1; x <= p2.x; x++)
		{
			if (d > 0)
			{
				y += stp;
				d += d2;
			}
			else
				d += d1;

			drawPixel(x, y, polySize, polygon, TRIANG_COLOR);
		}
	}
	else
	{
		if (dx < 0)
		{
			stp = -1;
			dx = -dx;
		}
		d = (dx * 2) - dy;
		d1 = dx * 2;
		d2 = (dx - dy) * 2;
		x = p1.x;

		for (y = p1.y + 1; y <= p2.y; y++)
		{
			if (d > 0)
			{
				x += stp;
				d += d2;
			}
			else
				d += d1;

			drawPixel(x, y, polySize, polygon, TRIANG_COLOR);
		}
	}
}

