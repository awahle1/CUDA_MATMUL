#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {

    float accum[TM] = {0.0};

    
    int outTileRow = threadIdx.x / BN;
    int outTileCol = threadIdx.x % BN;

    // This kernel is called so that blocks have the same number of threads as A and B tiles
    // In a more general version, each thread may have to grab multiple values
    int AtileRow = threadIdx.x / BK;
    int AtileCol = threadIdx.x % BK;
    int BtileRow = threadIdx.x / BN;
    int BtileCol = threadIdx.x % BN;

    __shared__ float A_s[BM][BK];
    __shared__ float B_s[BK][BN];

    for (int phase=0; phase<CEIL_DIV(K,BK); ++phase){
      // A col: BK*phase+ AtileCol, A row: AtileRow + blockIdx.y*BM
      A_s[AtileRow][AtileCol] = A[K*(AtileRow + blockIdx.y*BM) + (BK*phase+ AtileCol)];

      // B col: blockIdx.x*BN + BtileCol, B row: BtileRow + phase*BK
      B_s[BtileRow][BtileCol] = B[N*(BtileRow + phase*BK) + (blockIdx.x*BN + BtileCol)];
      __syncthreads();

      for (int dotIdx = 0; dotIdx<BK; ++dotIdx){
        float Btmp = B_s[dotIdx][BtileCol];
        for (int rowOffset=0; rowOffset<TM; ++ rowOffset){
          //BN threads per row, BM/TM rows, so the localBRow is the output tile row offset
          accum[rowOffset] += A_s[rowOffset + TM*outTileRow][dotIdx] * Btmp;
        }
      }

      __syncthreads();

    }
    for (int i=0; i<TM; ++i){
      //C col: blockIdx.x*BN + outTileCol, C row: blockIdx.y*BM + TM*outTileRow +i
      C[N*(blockIdx.y*BM + TM*outTileRow +i) + blockIdx.x*BN + outTileCol] = accum[i]*alpha + C[N*(blockIdx.y*BM + TM*outTileRow +i) + blockIdx.x*BN + outTileCol]*beta; 
    }
  }



// Indexing sanity checks:
// cols use blockIdx.x, rows use blockIdx.y
// For A, phase only affects col
// For B, phase only row
// A row indexes multiplied by K, Bs by N
