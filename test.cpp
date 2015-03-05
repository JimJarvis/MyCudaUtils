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

}