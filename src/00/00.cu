/**
    @brief Compare vector sum calculation functions in CPU vs GPU.
    @file 00.cu
    @author isquicha
    @version 0.1.0
*/

#include <stdio.h>
#include <time.h>

// Cuda headers are on CUDA Toolkit instalation path/VERSION/include
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N_THREADS 1024
#define LOOP_TIMES 10000000
/*
    The functions run too fast to see the difference, so to compare
    we run then in a loop.
    In my environment
*/

/**
 * Sum numbers of two vectors on GPU.
 *
 * The result is stored on a third vector.
 *
 * @param A First input vector pointer.
 * @param B Second input vector pointer.
 * @param C Output vector pointer.
 * @return void
 */
__global__ void dMatAdd(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    for (long j = 0; j < LOOP_TIMES; j++)
        C[i] = A[i] + B[i];
}

/**
 * Sum numbers of two vectors on CPU.
 *
 * The result is stored on a third vector.
 *
 * @param A First input vector pointer.
 * @param B Second input vector pointer.
 * @param C Output vector pointer.
 * @return void
 */
void hMatAdd(float *A, float *B, float *C)
{
    for (int i = 0; i < N_THREADS; i++)
    {
        for (long j = 0; j < LOOP_TIMES; j++)
        {
            C[i] = A[i] + B[i];
        }
    }
}

/**
 * Compare two float vectors
 *
 * @param A First input vector pointer.
 * @param B Second input vector pointer.
 * @return true if vectors are equal, false otherwise
 */
bool compare(float *A, float *B)
{
    for (int i = 0; i < N_THREADS; i++)
    {
        if (A[i] != B[i])
        {
            printf(
                "Elements are not equal. Index %d\t\tA: %f\t\tB:%f\n",
                i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char const *argv[])
{
    // Timing variables
    clock_t h_begin, h_end, d_begin, d_end;
    double h_time, d_time;

    // Size auxiliar
    size_t size = N_THREADS * sizeof(float);

    // Host memory allocation
    float *hA = (float *)malloc(size);
    float *hB = (float *)malloc(size);
    float *hC = (float *)malloc(size);
    float *hC2 = (float *)malloc(size);
    if (hA == NULL || hB == NULL || hC == NULL || hC2 == NULL)
    {
        printf("Malloc error!\n");
        exit(1);
    }

    // Device memory allocation
    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    // Vectors initialization with some values
    for (int i = 0, j = N_THREADS; i < N_THREADS; i++, j--)
    {
        hA[i] = float(i);
        //printf("hA[%d] = %f\n", i, hA[i]);
        hB[i] = float(j);
    }

    // Host function
    printf("Running Host function\n");
    h_begin = clock();
    hMatAdd(hA, hB, hC);
    h_end = clock();

    // Device function
    printf("Running Device function\n");
    d_begin = clock();
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    dMatAdd<<<1, N_THREADS>>>(dA, dB, dC);
    cudaMemcpy(hC2, dC, size, cudaMemcpyDeviceToHost);
    d_end = clock();

    // Results
    h_time = (double)(h_end - h_begin) / CLOCKS_PER_SEC;
    d_time = (double)(d_end - d_begin) / CLOCKS_PER_SEC;

    printf("Running Compare function\n");
    printf("Vectors are equal?: %s\n", compare(hC, hC2) ? "true" : "false");
    printf("CPU: %f seconds\n", h_time);
    printf("GPU: %f seconds\n", d_time);

    // Memory free
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    free(hC2);

    return 0;
}
