#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, float* A, float*B, float beta, float*C){
    int out_col = (threadIdx.x % BLOCKSIZE) + (blockIdx.x*BLOCKSIZE);
    int out_row = (threadIdx.x / BLOCKSIZE) + (blockIdx.y*BLOCKSIZE);

    if (out_row >= M || out_col >= N){
        return;
    }

    float accum = 0;

    for(int k = 0; k<K; ++k){
        accum += A[k + out_row * K]*B[out_col + k*N];
    }

    C[out_col + out_row*N] = alpha * accum + beta * C[out_col + out_row*N];
}