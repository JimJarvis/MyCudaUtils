#ifndef gpu_utils_h__
#define gpu_utils_h__

#include "myutils.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// To cheat the intellisense syntax highligher in VS2013
#ifdef __INTELLISENSE__
void __syncthreads();
void atomicAdd(uint*, uint);
#endif

#define CUDA_CHECK(ans) { gpuCheckError((ans), __FILE__, __LINE__); }
inline void gpuCheckError(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA_CHECK: %s in %s at %d\n", 
				cudaGetErrorString(code), file, line);
		exit(code);
	}
}

/*
class Timer
{
public:
	Timer() {};
	virtual ~Timer() {};
	virtual void start() = 0;
	virtual ulong elapsed() = 0;
	virtual void printElapsed(string msg = "")
	{
		std::cout << msg << ": " << elapsed() << " ms" << std::endl;
	}
};

class GpuTimer : public Timer
{
public:
	GpuTimer()
	{
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
	}

	~GpuTimer()
	{
		cudaEventDestroy(startTime);
		cudaEventDestroy(stopTime);
	}

	void start()
	{
		cudaEventRecord(startTime, 0);
	}

	ulong elapsed()
	{
		cudaEventRecord(stopTime, 0);

		float elapsed;
		cudaEventSynchronize(stopTime);
		cudaEventElapsedTime(&elapsed, startTime, stopTime);
		return (ulong) elapsed;
	}

private:
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
};

class CpuTimer : public Timer
{
public:
	CpuTimer() : startTime(0), stopTime(0)
	{
	}

	void start()
	{
		startTime = clock();
	}

	ulong elapsed()
	{
		stopTime = clock();
#ifdef WIN32
		return stopTime - startTime;
#else
		// ugly workaround for unix: how can it be DIFFERENT??!!
		return (stopTime - startTime) / 1000;
#endif
	}

private:
	clock_t startTime;
	clock_t stopTime;
};

template<typename T>
class FileReader
{
public:
	// ugly workaround for non C++11 CUDA 5.5
	FileReader(char *filename)
		: ifs(filename), sepLeng(0)
	{ 
		if (!ifs.is_open())
			cout << "FileReader: " + string(filename) + " error" << endl;
	}

	~FileReader()
	{
		ifs.close();
	}

	FileReader& setSeperatorLength(int sepLeng)
	{
		this->sepLeng = sepLeng;
		return *this;
	}

	T read()
	{
		T token;
		ifs >> token;
        char crap; // discard
        int len = this->sepLeng;
		while (--len >= 0 && ifs >> crap);
		return token;
	}

	vector<T> toVector(int size = -3)
	{
		vector<T> vec;
		T token;
		while (ifs >> token && (size == -3 || size-- > 0))
		{
			vec.push_back(token);
            char crap; // discard
            int len = this->sepLeng;
            while (--len >= 0 && ifs >> crap);
		}
		return vec;
	}

private:
	ifstream ifs;
	int sepLeng;
};
*/

/**************************************
************ CPU - GPU operations **************
**************************************/
template<typename T>
void gMemcpyVectorToDevice(const vector<T>& h_data, T *d_data)
{
	CUDA_CHECK(
			cudaMemcpy((void *)d_data,
			(void *) &h_data[0],
			h_data.size() * sizeof(T),
			cudaMemcpyHostToDevice)
	);
}

template<typename T>
void gMemcpyHostToDevice(T *h_data, T *d_data, size_t size)
{
	CUDA_CHECK(
			cudaMemcpy((void *)d_data,
			(void *) h_data,
			size * sizeof(T),
			cudaMemcpyHostToDevice)
	);
}

template<typename T>
void gMemcpyDeviceToVector(T *d_data, const vector<T>& h_data)
{
	CUDA_CHECK(
		cudaMemcpy((void *)&h_data[0],
			(void *)d_data,
			h_data.size() * sizeof(T),
			cudaMemcpyDeviceToHost)
	);
}

template<typename T>
void gMemcpyDeviceToHost(T *d_data, T *h_data, size_t size)
{
	CUDA_CHECK(
		cudaMemcpy((void *)h_data,
			(void *)d_data,
			size * sizeof(T),
			cudaMemcpyDeviceToHost)
	);
}

template<typename T>
T *gMalloc(size_t size)
{
	T* d_data;
	CUDA_CHECK(
		cudaMalloc((void **)&d_data, size * sizeof(T))
	);
	return d_data;
}

template<typename T>
void gMemset(T *d_data, size_t size, T value = 0)
{
	CUDA_CHECK(cudaMemset((void *)d_data, value, size * sizeof(T)));
}

template<typename T>
void gFree(T *d_data)
{
	CUDA_CHECK(cudaFree(d_data));
}

#endif // gpu_utils_h__