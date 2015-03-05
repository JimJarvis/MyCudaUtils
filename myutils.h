/*
 * myutils.h
 * (c) 2015
 * Author: Jim Fan
 * Common C++ header inclusion and print/vector utils
 */
#ifndef MYUTILS_H_
#define MYUTILS_H_

#include <iostream>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <memory>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
typedef unsigned long ulong;
typedef unsigned int uint;

/**************************************
************ Printing **************
**************************************/
template<typename Container>
string container2str(Container& vec,
		string leftDelimiter="[", string rightDelimiter="]")
{
	using ElemType = typename Container::value_type;
	std::ostringstream oss;
	oss << leftDelimiter;
//	for (ElemType& ele : vec)
//		oss << ele << ", ";
	for (int i = 0; i < vec.size(); ++i)
		oss << vec[i] << ", ";
	string s = oss.str();
	return (s.size() > leftDelimiter.size() ?
			s.substr(0, s.size() - 2) : s) + rightDelimiter;
}
template<typename Container>
string container2str(Container&& vec,
		string leftDelimiter="[", string rightDelimiter="]")
{
	return container2str(vec, leftDelimiter, rightDelimiter);
}

template<typename T>
std::ostream& operator<<(std::ostream& oss, vector<T>& vec)
{
	return oss << container2str(vec);
}
template<typename T>
std::ostream& operator<<(std::ostream& oss, vector<T>&& vec)
{
	return oss << vec;
}

// print basic array
template <typename T>
void printArray(T *arr, int size)
{
	cout << "[";
	int i;
	for (i = 0; i < size - 1; ++i)
		cout << arr[i] << ", ";
	cout << arr[i] << "]\n";
}

/**************************************
************ Misc **************
**************************************/
void myassert(bool cond, string errmsg = "")
{
	if (!cond)
	{
		cerr << "[Assert Fail] " <<  errmsg << endl;
		exit(1);
	}
}
#endif /* MYUTILS_H_ */