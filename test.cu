/*
 * test.cpp
 * (c) 2015
 * Author: Jim Fan
 * See below link for how to support C++11 in eclipse
 * http://scrupulousabstractions.tumblr.com/post/36441490955/eclipse-mingw-builds
 */
#include "mytimer.h"

int main(int argc, char **argv)
{
#ifdef is_CPP_11
	cout << "C++ 11 supported!" << endl;
#else
	cout << "C++ 11 NOT supported!" << endl;
#endif

#ifdef is_CUDA
	cout << "CUDA supported!" << endl;
#else
	cout << "CUDA NOT supported!" << endl;
#endif


	double p = 66;
	GpuTimer t;
	t.start();
	for (int i = 0; i < 10000000; ++i)
		p += i / p - std::sqrt(p);

	t.setResolution(Timer::Millisec).printElapsed();

	vector<int> a;
	a.push_back(5); a.push_back(1); a.push_back(4); a.push_back(9); a.push_back(0);
	cout << a << endl;
	cout << p << endl;
}
