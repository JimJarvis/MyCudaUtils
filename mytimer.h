/*
 * mytimer.h
 * (c) 2015
 * Author: Jim Fan
 */
#ifndef MYTIMER_H_
#define MYTIMER_H_

#include "myutils.h"
#include <ctime>
#ifdef CPP_11
#include <chrono>
#endif

class Timer
{
public:
	enum Resolution
	{
		Sec, Millisec, Microsec
	};

	Timer(Resolution scale) : scale(scale)
	{ }

	virtual ~Timer() {};

	virtual void start() = 0;

	ulong elapsed()
	{
		return std::round(_elapsed_sec() * (scale == Resolution::Sec ? 1 :
					scale == Resolution::Millisec ? 1000 :
					scale == Resolution::Microsec ? 1e6 : 1));
	}

	void printElapsed(string msg = "")
	{
		if (msg != "")
			msg += ": ";
		string scaleName = scale == Resolution::Sec ? "seconds" :
					scale == Resolution::Millisec ? "milliseconds" :
					scale == Resolution::Microsec ? "microseconds" : "";

		cout << msg << elapsed() << " " << scaleName << " elapsed" << endl;
	}

	Timer& setResolution(Resolution scale)
	{
		this->scale = scale;
		return *this;
	}

	virtual void showTime() {};

protected:
	// must return in seconds
	virtual double _elapsed_sec() = 0;

	Resolution scale;
};

#ifdef CPP_11
class CpuTimer : public Timer
{
public:
	CpuTimer(Resolution scale = Resolution::Millisec) : Timer(scale)
	{ }

	~CpuTimer() {}

	void start()
	{
		start_time = std::chrono::system_clock::now();
	}

	void showTime()
	{
		time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		cout << ctime(&t) << endl;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;

protected:
	double _elapsed_sec()
	{
		end_time = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_time - start_time;
		return elapsed_seconds.count();
	}
};

#else // doesn't support C++11
class CpuTimer : public Timer
{
public:
	CpuTimer(Resolution scale = Resolution::Millisec) :
		Timer(scale), startTime(0), stopTime(0)
	{
	}

	~CpuTimer() {}

	void start()
	{
		startTime = clock();
	}

private:
	clock_t startTime, stopTime;

protected:
	double _elapsed_sec()
	{
		stopTime = clock();
		return (double)(stopTime - startTime) / CLOCKS_PER_SEC;
	}
};

#endif

#endif /* MYTIMER_H_ */
