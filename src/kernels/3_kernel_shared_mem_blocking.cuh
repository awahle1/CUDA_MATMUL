#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) 
{
    int localCol = threadIdx.x % BLOCKSIZE;
    int localRow = threadIdx.x / BLOCKSIZE;
    int outCol = blockIdx.x * BLOCKSIZE + localCol;
    int outRow = blockIdx.y * BLOCKSIZE + localRow;

    float accum = 0;

    __shared__ float A_s[BLOCKSIZE][BLOCKSIZE];
    __shared__ float B_s[BLOCKSIZE][BLOCKSIZE];

    for (int phase=0; phase<CEIL_DIV(K,BLOCKSIZE); ++phase){
        if ((BLOCKSIZE*blockIdx.y+localRow) < M && BLOCKSIZE*phase+localCol < K){
          A_s[localRow][localCol] = A[K*(BLOCKSIZE*blockIdx.y+localRow) + (BLOCKSIZE*phase+localCol)];
        }
        if((BLOCKSIZE*phase+localRow) < K && (BLOCKSIZE*blockIdx.x+localCol) < N){
          B_s[localRow][localCol] = B[N*(BLOCKSIZE*phase+localRow) + (BLOCKSIZE*blockIdx.x+localCol)];
        }
        __syncthreads();

        for (int k=0; k<BLOCKSIZE; ++k){
            accum += A_s[localRow][k] * B_s[k][localCol];
        }
        __syncthreads();
    }
    C[outRow*N + outCol] = alpha*accum + beta*C[outRow*N + outCol]; 
}