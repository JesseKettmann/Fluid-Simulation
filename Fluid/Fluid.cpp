#include "Fluid.h"
#include <cmath>	// std::floor
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

Fluid::Fluid()
	:
	s{0.0f},
	density{0.0f},
	Vx{0.0f},
	Vy{0.0f},
	Vx0{0.0f},
	Vy0{0.0f}
{
#if 0
	this->s = new float[N * N];
	this->density = new float[N * N];

	this->Vx = new float[N * N];
	this->Vy = new float[N * N];

	this->Vx0 = new float[N * N];
	this->Vy0 = new float[N * N];

	assert(s && density && Vx && Vy && Vx0 && Vy0);

	memset(s, 0, N * N);
	memset(density, 0, N * N);
	memset(Vx, 0, N * N);
	memset(Vy, 0, N * N);
	memset(Vx0, 0, N * N);
	memset(Vy0, 0, N * N);
#endif
}

void Fluid::Update() noexcept
{
	// Create the two input vectors
	int i;
	const int LIST_SIZE = 1024;
	int* A = (int*)malloc(sizeof(int) * LIST_SIZE);
	int* B = (int*)malloc(sizeof(int) * LIST_SIZE);
	for (i = 0; i < LIST_SIZE; i++) {
		A[i] = i;
		B[i] = LIST_SIZE - i;
	}

	// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;
#pragma warning (disable : 4996)
	fp = fopen("program.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
		&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		LIST_SIZE * sizeof(int), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char**)&source_str, (const size_t*)&source_size, &ret);

	int x = 1;

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	int y = 2;

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);

	// Execute the OpenCL kernel on the list
	size_t global_item_size = LIST_SIZE; // Process the entire lists
	size_t local_item_size = 64; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);

	// Read the memory buffer C on the device to the local variable C
	int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
		LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

	// Display the result to the screen
	for (i = 0; i < LIST_SIZE; i++)
		printf("%d + %d = %d\n", A[i], B[i], C[i]);

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	//return 0;

	Diffuse(1, Vx0.data(), Vx.data(), VISCOSITY, MOTION_SPEED);
	Diffuse(2, Vy0.data(), Vy.data(), VISCOSITY, MOTION_SPEED);

	Project(Vx0.data(), Vy0.data(), Vx.data(), Vy.data());

	Advect(1, Vx.data(), Vx0.data(), Vx0.data(), Vy0.data(), MOTION_SPEED);
	Advect(2, Vy.data(), Vy0.data(), Vx0.data(), Vy0.data(), MOTION_SPEED);

	Project(Vx.data(), Vy.data(), Vx0.data(), Vy0.data());

	Diffuse(0, s.data(), density.data(), DIFFUSION, MOTION_SPEED);
	Advect(0, density.data(), s.data(), Vx.data(), Vy.data(), MOTION_SPEED);
}

void Fluid::AddDensity(int x, int y, float amount) noexcept
{
	this->density[IX(x, y)] += amount;
}

void Fluid::AddVelocity(int x, int y, float amountX, float amountY) noexcept
{
	const int index = IX(x, y);

	this->Vx[index] += amountX;
	this->Vy[index] += amountY;
}

void Fluid::Diffuse(int b, float* x, float* x0, float diff, float dt) noexcept
{
	const float a = dt * diff * (N - 2) * (N - 2);
	LinearSolve(b, x, x0, a, 1 + SCALE * a);
}

void Fluid::LinearSolve(int b, float* x, float* x0, float a, float c) noexcept
{
	const float cRecip = 1.0f / c;
	for (int k = 0; k < ITERATIONS; k++)
	{
		for (int j = 1; j < N - 1; j++)
		{
			for (int i = 1; i < N - 1; i++)
			{
				x[IX(i, j)] =
					(x0[IX(i, j)]
						+ a * (x[IX(i + 1, j)]
							+ x[IX(i - 1, j)]
							+ x[IX(i, j + 1)]
							+ x[IX(i, j - 1)]
							)) * cRecip;
			}
		}

		SetBoundary(b, x);
	}
}

void Fluid::SetBoundary(int b, float* x) noexcept
{
	for (int i = 1; i < N - 1; i++)
	{
		x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
		x[IX(i, N - 1)] = b == 2 ? -x[IX(i, N - 2)] : x[IX(i, N - 2)];
	}
	for (int j = 1; j < N - 1; j++)
	{
		x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
		x[IX(N - 1, j)] = b == 1 ? -x[IX(N - 2, j)] : x[IX(N - 2, j)];
	}

	x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N - 1)] = 0.5f * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
	x[IX(N - 1, 0)] = 0.5f * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
	x[IX(N - 1, N - 1)] = 0.5f * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);

}

void Fluid::Project(float* velocX, float* velocY, float* p, float* div) noexcept
{
	for (int j = 1; j < N - 1; j++) {
		for (int i = 1; i < N - 1; i++) {
			div[IX(i, j)] = -0.5f * (
				velocX[IX(i + 1, j)]
				- velocX[IX(i - 1, j)]
				+ velocY[IX(i, j + 1)]
				- velocY[IX(i, j - 1)]
				) / N;
			p[IX(i, j)] = 0;
		}
	}

	SetBoundary(0, div);
	SetBoundary(0, p);
	LinearSolve(0, p, div, 1, 4);

	for (int j = 1; j < N - 1; j++) {
		for (int i = 1; i < N - 1; i++) {
			velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
				- p[IX(i - 1, j)]) * N;
			velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
				- p[IX(i, j - 1)]) * N;
		}
	}
	SetBoundary(1, velocX);
	SetBoundary(2, velocY);

}

void Fluid::Advect(int b, float* d, float* d0, float* velocX, float* velocY, float dt) noexcept
{ 
	float i0, i1, j0, j1;

	const float dtx = dt * (N - 2);
	const float dty = dt * (N - 2);

	float s0, s1, t0, t1;
	float tmp1, tmp2, x, y;

	constexpr float Nfloat = static_cast<float>(N);
	float ifloat, jfloat;
	int i, j;

	for (j = 1, jfloat = 1; j < N - 1; j++, jfloat++)
	{
		for (i = 1, ifloat = 1; i < N - 1; i++, ifloat++)
		{
			const int index = IX(i, j);

			tmp1 = dtx * velocX[index];
			tmp2 = dty * velocY[index];

			x = ifloat - tmp1;
			y = jfloat - tmp2;

			if (x < 0.5f) x = 0.5f;
			if (x > Nfloat + 0.5f) x = Nfloat + 0.5f;
			i0 = std::floor(x);
			i1 = i0 + 1.0f;
			if (y < 0.5f) y = 0.5f;
			if (y > Nfloat + 0.5f) y = Nfloat + 0.5f;
			j0 = std::floor(y);
			j1 = j0 + 1.0f;


			s1 = x - i0;
			s0 = 1.0f - s1;
			t1 = y - j0;
			t0 = 1.0f - t1;


			int i0i = static_cast<int>(i0);
			int i1i = static_cast<int>(i1);
			int j0i = static_cast<int>(j0);
			int j1i = static_cast<int>(j1);

			d[index] =
				s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)]) +
				s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)]);

		}
	}
	SetBoundary(b, d);
}

Fluid::~Fluid(){}