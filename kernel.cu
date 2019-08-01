
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <utility>

constexpr int kb=1<<10;
constexpr int bytes=24;
constexpr int size_per_thread=bytes*kb/(512*sizeof(int));

namespace kernel
{
__global__
void global_to_shared(int *global)
{
	__shared__ int shared[bytes*kb/sizeof(int)];
	__syncthreads();

#pragma unroll 12
	for(int i=0; i<size_per_thread; ++i)
	{
		shared[threadIdx.x*size_per_thread+i]=global[threadIdx.x*size_per_thread+i];
	}
}


__global__
void shared_to_global(int *global)
{
	__shared__ int shared[bytes*kb/sizeof(int)];
	__syncthreads();

#pragma unroll 12
	for(int i=0; i<size_per_thread; ++i)
	{
		global[threadIdx.x*size_per_thread+i]=shared[threadIdx.x*size_per_thread+i];
	}
}

}


auto measure_global_shared()
{
	constexpr int repeat=10000;

	float gs_sum=0, sg_sum=0, time;

	int *global;
	cudaMalloc((void **)&global, bytes*kb);

	const dim3 grid(1);
	const dim3 block(512);

	for(int i=0; i<repeat; ++i)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaEventRecord(start);
		kernel::global_to_shared<<<grid, block>>>(global);
		cudaEventRecord(stop);
		cudaEventElapsedTime(&time, start, stop);
		gs_sum+=time;


		cudaEventRecord(start);
		kernel::shared_to_global<<<grid, block>>>(global);
		cudaEventRecord(stop);
		cudaEventElapsedTime(&time, start, stop);
		sg_sum+=time;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	cudaFree(global);

	return std::make_pair(gs_sum, sg_sum);
}


int main()
{
	std::cout<<"data size[KB], global to shared[ms], shared to global[ms]"<<std::endl;
	const auto time=measure_global_shared();
	std::cout<<bytes<<","<<time.first<<","<<time.second<<std::endl;
}
