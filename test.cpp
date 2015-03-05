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
#if __cplusplus > 201100L
	cout << "C++ 11 supported!" << endl;
#else
	cout << "C++ 11 NOT supported!" << endl;
#endif


	double p = 66;
	CpuTimer t;
	t.start();
	for (int i = 0; i < 10000000; ++i)
		p += i / p - std::sqrt(p);

	t.setResolution(Timer::Millisec).printElapsed();

	vector<int> a {1, 2, 3, 4, 5};
	cout << a << endl;
	cout << p << endl;
}
