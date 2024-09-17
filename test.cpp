// myparallel.cpp : This file contains small parallel program to compute multiplication of vectors by exploiting Accelerator like GPU
//

#include <iostream>
#include <omp.h>

const int THRESHOLD = 700;


void init(int* a, int* b, size_t size) 
{
    int i = 0;
#pragma omp parallel for simd
    for (i = 0; i < size; i++) {
        a[i] = i * 2;
        b[i] = i * 3;
    }
}

int main()
{
    int target_dev_count = omp_get_num_devices();
    std::cout << "Started parallel program with target devoices: .."<< target_dev_count<< std::endl;
    int x[800], y[800], z[800];


    int i, dimension = 800;  // in a larger program this could be passed in as an argument in function call chain
    init(x, y, dimension);

#pragma omp target data if (dimension > THRESHOLD) map(to:x[0:dimension],y[0:dimension]) map(from:z[0:dimension])
    #pragma omp teams distribute parallel for simd 
    for(i=0; i<dimension; i++)
       z[i]= x[i]*y[i];

#pragma omp target update from(z[0:dimension])

    std::cout << "a few  Compiuted values are.. "<< std::endl;
    for (i = 0; i < 64 && i < dimension; i++)
        std::cout <<  z[i]  << " ";

    std::cout << "Thank You!" << std::endl;
}
