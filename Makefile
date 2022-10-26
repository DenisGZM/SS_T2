monte-carlo-d: main.cpp
	mpicxx -O2 -o monte-carlo-d main.cpp -std=c++14 -DDISTRIBUTED
monte-carlo-m:
	mpicxx -O2 -o monte-carlo-m main.cpp -std=c++14 -DMASTER

run-d: monte-carlo-d
	mpiexec -n 4 ./monte-carlo-d 1e-5
	