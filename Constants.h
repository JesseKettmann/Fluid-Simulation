#pragma once
#include "Engine/olcPixelGameEngine.h"
#include <algorithm> // std::clamp
#include <cassert>
#include <memory>
#include <utility>
#include <random>

namespace
{

#define SCALE			3	
#define N				160 // Width/height of screen
#define ITERATIONS		16
#define VISCOSITY		0.0000001f
#define DIFFUSION		0.0f
#define MOTION_SPEED	0.2f

	//static constexpr const int SCALE = 3;
	//static constexpr const int N = 160;
	//static constexpr const int ITERATIONS = 16;

	//static constexpr const float VESCOSITY = 0.0000001f; // thickness of fluid
	//static constexpr const float DIFFUSION = 0.0f;
	//static constexpr const float MOTION_SPEED = 0.2f;

	/**
	* Converts 2D coords into 1D ( x,y into index )
	*/
	template<class T>
	static constexpr T IX(T x, T y) noexcept
	{
		x = std::clamp(x, 0, N - 1);
		y = std::clamp(y, 0, N - 1);
		return x + (y * N);
	}
}
