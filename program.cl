#include "Constants.h"

__kernel void HelloWorld(__global char* data, __global int* number)
{
	data[0] = 'H';
	data[1] = 'e';
	data[2] = 'l';
	data[3] = 'l';
	data[4] = 'o';
	data[5] = 'W';
	data[6] = 'o';
	data[7] = 'r';
	data[8] = 'l';
	data[9] = 'd';
	data[10] = '\n';

	int i = get_global_id(0);
	int num = number[i];
	number[i] = i;

}

__kernel void LinearSolve(__global float* x, __global float* x0, const float a, const float cRecip)
{
	int i = get_global_id(0);
	i = i + N + 1 + i / (N - 2) * 2; //Skip all perimeter cells
	x[i] = (x0[i] + a * (x[i-1] + x[i+1] + x[i+N] + x[i-N])) * cRecip;
}

__kernel void SetBoundaryHorizontal(__global float* x, const int b) {
    int i = get_global_id(0);
    x[i + 1] = b == 2 ? -x[i + 1 + N] : x[i + 1 + N];
    x[N * N - i - 2] = b == 2 ? -x[N * (N - 1) - i - 2] : x[N * (N - 1) - i - 2];
}

__kernel void SetBoundaryVertical(__global float* x, const int b) {
    int j = get_global_id(0);
    x[N * (j + 1)] = b == 1 ? -x[N * (j + 1) + 1] : x[N * (j + 1) + 1];
    x[N * (j + 2) - 1] = b == 1 ? -x[N * (j + 2) - 2] : x[N * (j + 2) - 2];
}

__kernel void SetCorners(__global float* x) {
    x[0] = 0.5f * (x[1] + x[N]);
    x[N - 1] = 0.5f * (x[N - 2] + x[2 * N - 1]);
    x[(N - 1) * N] = 0.5f * (x[(N - 1) * N + 1] + x[(N - 2) * N]);
    x[N * N - 1] = 0.5f * (x[N * N - 2] + x[N * (N - 1)]);
}

__kernel void Project1(__global float* velocX, __global float* velocY, __global float* p, __global float* div)
{
	int i = get_global_id(0);
	i = i + N + 1 + i / (N - 2) * 2; //Skip all perimeter cells
	div[i] = -0.5f * (velocX[i+1] - velocX[i-1] + velocY[i+N] - velocY[i-N]) / N;
	p[i] = 0;
}

__kernel void Project2(__global float* velocX, __global float* velocY, __global float* p, __global float* div)
{
	int i = get_global_id(0);
	i = i + N + 1 + i / (N - 2) * 2; //Skip all perimeter cells
	velocX[i] -= 0.5f * (p[i+1] - p[i-1]) * N;
	velocY[i] -= 0.5f * (p[i+N] - p[i-N]) * N;
}

__kernel void Advect(const int b, __global float* d, __global float* d0, __global float* velocX, __global float* velocY, const float dtx, const float dty)
{
	int i = get_global_id(0);
	i = i + N + 1 + i / (N - 2) * 2; //Skip all perimeter cells

	float tmp1 = dtx * velocX[i];
	float tmp2 = dty * velocY[i];

	//I'm not sure how to continue want floats vs ints en drama
	float x = (i % N) - tmp1;
	float y = (i / N) - tmp2;

	x = clamp(x, 0.5f, 0.5f + N);
	y = clamp(y, 0.5f, 0.5f + N);

	int i0 = (int)x;
	int j0 = (int)y;

	float s1 = x - i0;
	float s0 = 1.0f - s1;
	float t1 = y - j0;
	float t0 = 1.0f - t1;

	d[i] =
		s0 * (t0 * d0[i0 + j0 * N] + t1 * d0[i0 + (j0 + 1) * N]) +
		s1 * (t0 * d0[(i0 + 1) + j0 * N] + t1 * d0[(i0 + 1) + (j0 + 1) * N]);
	
}