#include "Fluid.h"
#include <cmath>	// std::floor
#include <CL/cl.h>
#include <fstream>
#include <iostream>
using namespace std;

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

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); //maybe change to TYPE_ALL if necessary

	device = devices.front();

	std::ifstream programFile("program.cl");
	std::string src(std::istreambuf_iterator<char>(programFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	context = cl::Context(device);
	program = cl::Program(context, sources);
	auto err = program.build();
	if (err) cout << err << endl;
	
	queue = cl::CommandQueue(context, device);
}

void Fluid::Update() noexcept
{

	//char buf[16] = { 0 };
	//int numbers[] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//cl::Buffer memBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(buf), buf); //Change read/write rights if necessary
	//cl::Buffer memNum(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(numbers), numbers); //Change read/write rights if necessary
	//cl::Kernel kernel(program, "HelloWorld");
	//kernel.setArg(0, memBuf);
	//kernel.setArg(1, memNum);

	//cl::CommandQueue queue(context, device);
	//queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(17));
	//queue.finish();
	//queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);
	//queue.enqueueReadBuffer(memNum, CL_TRUE, 0, sizeof(numbers), numbers);

	//int length = sizeof(numbers) / sizeof(int);
	//for (int i = 0; i < length; i++)
	//{
	//	cout << numbers[i] << endl;
	//}
	//cout << buf << endl;

	// Create buffers and transfer data to device
	cl::Buffer VxBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vx.data());
	cl::Buffer Vx0Buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vx0.data());
	cl::Buffer VyBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vy.data());
	cl::Buffer Vy0Buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vy0.data());

	Diffuse(1, Vx0Buf, VxBuf, VISCOSITY, MOTION_SPEED);
	Diffuse(2, Vy0Buf, VyBuf, VISCOSITY, MOTION_SPEED);

	Project(Vx0Buf, Vy0Buf, VxBuf, VyBuf);

	// Read back results
	queue.enqueueReadBuffer(VxBuf, CL_TRUE, 0, size_t(N * N * 4), Vx.data());
	queue.enqueueReadBuffer(Vx0Buf, CL_TRUE, 0, size_t(N * N * 4), Vx0.data());
	queue.enqueueReadBuffer(VyBuf, CL_TRUE, 0, size_t(N * N * 4), Vy.data());
	queue.enqueueReadBuffer(Vy0Buf, CL_TRUE, 0, size_t(N * N * 4), Vy0.data());

	Advect(1, Vx.data(), Vx0.data(), Vx0.data(), Vy0.data(), MOTION_SPEED);
	Advect(2, Vy.data(), Vy0.data(), Vx0.data(), Vy0.data(), MOTION_SPEED);

	queue.enqueueWriteBuffer(VxBuf, CL_TRUE, 0, size_t(N * N * 4), Vx.data());
	queue.enqueueWriteBuffer(VyBuf, CL_TRUE, 0, size_t(N * N * 4), Vy.data());

	Project(VxBuf, VyBuf, Vx0Buf, Vy0Buf);

	// Read back results
	queue.enqueueReadBuffer(VyBuf, CL_TRUE, 0, size_t(N * N * 4), Vy.data());
	queue.enqueueReadBuffer(Vy0Buf, CL_TRUE, 0, size_t(N * N * 4), Vy0.data());
	queue.enqueueReadBuffer(VxBuf, CL_TRUE, 0, size_t(N * N * 4), Vx.data());
	queue.enqueueReadBuffer(Vx0Buf, CL_TRUE, 0, size_t(N * N * 4), Vx0.data());

	// Create buffers and transfer data to device
	cl::Buffer sBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), s);
	cl::Buffer densityBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), density);

	Diffuse(0, sBuf, densityBuf, DIFFUSION, MOTION_SPEED);

	// Read back results
	queue.enqueueReadBuffer(sBuf, CL_TRUE, 0, size_t(N * N * 4), s);

	Advect(0, density, s, Vx.data(), Vy.data(), MOTION_SPEED);

	queue.finish();
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

void Fluid::Diffuse(int b, cl::Buffer x, cl::Buffer x0, float diff, float dt) noexcept
{
	const float a = dt * diff * (N - 2) * (N - 2);
	LinearSolve(b, x, x0, a, 1 + SCALE * a);
}

void Fluid::LinearSolve(int b, cl::Buffer x, cl::Buffer x0, float a, float c) noexcept
{
	const float cRecip = 1.0f / c;
	for (int k = 0; k < ITERATIONS; k++)
	{
		// Create and set arguments for the linear solve kernel
		cl::Kernel LinearSolveKernel(program, "LinearSolve");
		LinearSolveKernel.setArg(0, x);
		LinearSolveKernel.setArg(1, x0);
		LinearSolveKernel.setArg(2, sizeof(float), &a);
		LinearSolveKernel.setArg(3, sizeof(float), &cRecip);

		// Enqueue kernel
		queue.enqueueNDRangeKernel(LinearSolveKernel, cl::NullRange, cl::NDRange(N * N - (4 * N - 4)));

		SetBoundary(b, x);
	}
}

void Fluid::SetBoundaryOld(int b, float* x) noexcept
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

void Fluid::SetBoundary(int b, cl::Buffer x) noexcept
{
	// Create and set arguments for the horizontal boundary kernel
	cl::Kernel horizontalKernel(program, "SetBoundaryHorizontal");
	horizontalKernel.setArg(0, x);
	horizontalKernel.setArg(1, b);

	// Create and set arguments for the vertical boundary kernel
	cl::Kernel verticalKernel(program, "SetBoundaryVertical");
	verticalKernel.setArg(0, x);
	verticalKernel.setArg(1, b);

	// Create and set arguments for the corner kernel
	cl::Kernel cornerKernel(program, "SetCorners");
	cornerKernel.setArg(0, x);

	queue.enqueueNDRangeKernel(horizontalKernel, cl::NullRange, cl::NDRange(N - 2));
	queue.enqueueNDRangeKernel(verticalKernel, cl::NullRange, cl::NDRange(N - 2));
	queue.enqueueTask(cornerKernel);
}

void Fluid::Project(cl::Buffer velocX, cl::Buffer velocY, cl::Buffer p, cl::Buffer div) noexcept
{

#pragma region Project1_Kernelized
	// Create and set arguments for the corner kernel
	cl::Kernel project1Kernel(program, "Project1");

	project1Kernel.setArg(0, velocX);
	project1Kernel.setArg(1, velocY);
	project1Kernel.setArg(2, p);
	project1Kernel.setArg(3, div);
	queue.enqueueNDRangeKernel(project1Kernel, cl::NullRange, cl::NDRange(N * N - (4 * N - 4)));

	SetBoundary(0, div);
	SetBoundary(0, p);
#pragma endregion

	LinearSolve(0, p, div, 1, 4);

	//--Original code--:
	/*for (int j = 1; j < N - 1; j++) {
		for (int i = 1; i < N - 1; i++) {
			velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
				- p[IX(i - 1, j)]) * N;
			velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
				- p[IX(i, j - 1)]) * N;
		}
	}*/

#pragma region Project2_Kernelized
	// Create and set arguments for the corner kernel
	cl::Kernel project2Kernel(program, "Project2");

	project2Kernel.setArg(0, velocX);
	project2Kernel.setArg(1, velocY);
	project2Kernel.setArg(2, p);
	project2Kernel.setArg(3, div);
	queue.enqueueNDRangeKernel(project2Kernel, cl::NullRange, cl::NDRange(N * N - (4 * N - 4)));

	
#pragma endregion

	SetBoundary(1, velocX);
	SetBoundary(2, velocY);
}

void Fluid::ProjectOld(float* velocX, float* velocY, float* p, float* div) noexcept
{

#pragma region Project1_Kernelized
	// Create and set arguments for the corner kernel
	cl::Kernel project1Kernel(program, "Project1");

	//Remove this later once buffers have been hoisted
	cl::Buffer velocXBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), velocX);
	cl::Buffer velocYBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), velocY);
	cl::Buffer pBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), p);
	cl::Buffer divBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), div);

	project1Kernel.setArg(0, velocXBuf);
	project1Kernel.setArg(1, velocYBuf);
	project1Kernel.setArg(2, pBuf);
	project1Kernel.setArg(3, divBuf);
	queue.enqueueNDRangeKernel(project1Kernel, cl::NullRange, cl::NDRange(N * N - (4 * N - 4)));

	SetBoundary(0, divBuf);
	SetBoundary(0, pBuf);
#pragma endregion

	LinearSolve(0, pBuf, divBuf, 1, 4);

	// Read back results
	queue.enqueueReadBuffer(pBuf, CL_TRUE, 0, size_t(N * N * 4), p);

	//--Original code--:
	/*for (int j = 1; j < N - 1; j++) {
		for (int i = 1; i < N - 1; i++) {
			velocX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)]
				- p[IX(i - 1, j)]) * N;
			velocY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)]
				- p[IX(i, j - 1)]) * N;
		}
	}*/

#pragma region Project2_Kernelized
	// Create and set arguments for the corner kernel
	cl::Kernel project2Kernel(program, "Project2");

	//Remove this later once buffers have been hoisted
	pBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), p);
	divBuf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), div);

	project2Kernel.setArg(0, velocXBuf);
	project2Kernel.setArg(1, velocYBuf);
	project2Kernel.setArg(2, pBuf);
	project2Kernel.setArg(3, divBuf);
	queue.enqueueNDRangeKernel(project2Kernel, cl::NullRange, cl::NDRange(N * N - (4 * N - 4)));


#pragma endregion

	SetBoundary(1, velocXBuf);
	SetBoundary(2, velocYBuf);

	// Read back results
	queue.enqueueReadBuffer(velocXBuf, CL_TRUE, 0, size_t(N * N * 4), velocX);
	queue.enqueueReadBuffer(velocYBuf, CL_TRUE, 0, size_t(N * N * 4), velocY);
	queue.enqueueReadBuffer(pBuf, CL_TRUE, 0, size_t(N * N * 4), p);
	queue.enqueueReadBuffer(divBuf, CL_TRUE, 0, size_t(N * N * 4), div);
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
	SetBoundaryOld(b, d);
}

Fluid::~Fluid(){}