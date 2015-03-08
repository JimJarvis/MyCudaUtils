/*
 * Eona Studio (c) 2015
 * Author: Jim Fan
 */
#ifndef FILEUTILS_H_
#define FILEUTILS_H_

#include "myutils.h"
using std::ifstream;

template<typename T>
class FileReader
{
public:
#ifdef is_CPP_11
	FileReader(string filename)
#else
	//ugly workaround for non-c++11 CUDA 5.5
	FileReader(char *filename)
#endif
		: ifs(filename), sepLeng(0)
	{
		if (!ifs.is_open())
			cout << string("FileReader: ") + filename + " error" << endl;
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


#endif /* FILEUTILS_H_ */
