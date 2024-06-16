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
	// Create buffers and transfer data to device
	cl::Buffer VxBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vx.data());
	cl::Buffer Vx0Buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vx0.data());
	cl::Buffer VyBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vy.data());
	cl::Buffer Vy0Buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), Vy0.data());
	cl::Buffer sBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), s);
	cl::Buffer densityBuf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_t(N * N * 4), density);

	Diffuse(1, Vx0Buf, VxBuf, VISCOSITY, MOTION_SPEED);
	Diffuse(2, Vy0Buf, VyBuf, VISCOSITY, MOTION_SPEED);
	Project(Vx0Buf, Vy0Buf, VxBuf, VyBuf);
	Advect(1, VxBuf, Vx0Buf, Vx0Buf, Vy0Buf, MOTION_SPEED);
	Advect(2, VyBuf, Vy0Buf, Vx0Buf, Vy0Buf, MOTION_SPEED);
	Project(VxBuf, VyBuf, Vx0Buf, Vy0Buf);

	Diffuse(0, sBuf, densityBuf, DIFFUSION, MOTION_SPEED);
	Advect(0, densityBuf, sBuf, VxBuf, VyBuf, MOTION_SPEED);

	// Read back results
	queue.enqueueReadBuffer(VxBuf, CL_TRUE, 0, size_t(N * N * 4), Vx.data());
	queue.enqueueReadBuffer(Vx0Buf, CL_TRUE, 0, size_t(N * N * 4), Vx0.data());
	queue.enqueueReadBuffer(VyBuf, CL_TRUE, 0, size_t(N * N * 4), Vy.data());
	queue.enqueueReadBuffer(Vy0Buf, CL_TRUE, 0, size_t(N * N * 4), Vy0.data());
	queue.enqueueReadBuffer(sBuf, CL_TRUE, 0, size_t(N * N * 4), s);
	queue.enqueueReadBuffer(densityBuf, CL_TRUE, 0, size_t(N * N * 4), density);

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

void Fluid::Advect(int b, cl::Buffer d, cl::Buffer d0, cl::Buffer velocX, cl::Buffer velocY, float dt) noexcept
{ 
	const float dtx = dt * (N - 2);
	const float dty = dt * (N - 2);

	// Create and set arguments
	cl::Kernel advectKernel(program, "Advect");

	advectKernel.setArg(0, sizeof(int), &b);
	advectKernel.setArg(1, d);
	advectKernel.setArg(2, d0);
	advectKernel.setArg(3, velocX);
	advectKernel.setArg(4, velocY);
	advectKernel.setArg(5, sizeof(float), &dtx);
	advectKernel.setArg(6, sizeof(float), &dty);

	queue.enqueueNDRangeKernel(advectKernel, cl::NullRange, cl::NDRange(N * N - (4 * N - 4)));

	SetBoundary(b, d);
}

Fluid::~Fluid(){}