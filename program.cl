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
	number[i] = num + 3;

}

__kernel void LinearSolve(__global float* x, __global float* x0, const float a, const float cRecip)
{
	int i = get_global_id(0);
	//i = i + N + 1 + i / (N - 2) * 2;
	//x[i] = (x0[i] + a * (x[i-1] + x[i+1] + x[i+N] + x[i-N])) * cRecip;
	x[i] = 5000;
}