#========= Tool-chain config ==========
CC=g++
CCFLAGS=-std=c++11 -I. #-I/usr/include/OpenEXR
LDFLAGS= #-lImath -lIlmImf -lHalf
NVFLAGS=$(CCFLAGS) -arch=sm_20

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    NVCC=/Developer/NVIDIA/CUDA-7.0/bin/nvcc
else
    NVCC=nvcc
endif


#========= Source files config ==========
MAIN=test

HEADERS=myutils.h gputils.h mytimer.h fileutils.h
OBJS=$(MAIN).o

# $@ left hand side of ':'
# $^ right hand side of ':'
# $< first item in dependency list 

# handle all CUDA src
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# handle all regular c++ src
%.o: %.cpp $(HEADERS)
	$(CC) $(CCFLAGS) -c $< -o $@

# linking
$(MAIN): $(OBJS)
	$(NVCC) $^ -o $@ $(NVFLAGS) $(LDFLAGS)

#========= Phony targets ==========
.PHONY: clean
clean:
	@rm -f $(MAIN) *.o *.exe *.lib *.exp
	@echo cleaned

.PHONY: all
all: clean $(MAIN)
	@echo Successfully rebuilt

#==== Testing ====
TEST1=whatever_test_file
TEST2=whatever_test_file
TEST3=whatever_test_file

.PHONY: run
run: $(MAIN)
	./$(MAIN)

.PHONY: t1
t1: $(MAIN)
	./$(MAIN) $(TEST1)

.PHONY: t2
t2: $(MAIN)
	./$(MAIN) $(TEST2)

.PHONY: t3
t3: $(MAIN)
	./$(MAIN) $(TEST3)
