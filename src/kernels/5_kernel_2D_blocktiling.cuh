#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  
  int blockHeight = BM/TM;
  int blockWidth = BN/TN;


  //Layers of loops
  //Chunks of size BMxBK and BKxBN
  //Sub-Chunks

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int threadRow = threadIdx.x / blockWidth;
  int threadCol = threadIdx.x % blockWidth;

  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  float threadResults[TM][TN] = {0.0};

  int strideA = blockDim.x / BK;
  int strideB = blockDim.x / BN;

  int rowOffsetA = threadIdx.x / BK;
  int colOffsetA = threadIdx.x % BK;

  int rowOffsetB = threadIdx.x / BN;
  int colOffsetB = threadIdx.x % BN;

  // Chunks of size BMxBK and BKxBN
  for(int blockOffset = 0; blockOffset<K; blockOffset += BK){
    for (int loadInd = 0; loadInd<BM; loadInd += strideA){
      As[loadInd + rowOffsetA][colOffsetA] = A[(rowOffsetA + loadInd + blockIdx.y*BM)*K + blockOffset + colOffsetA];
    }
    for (int loadInd = 0; loadInd<BK; loadInd += strideB){
      Bs[loadInd + rowOffsetB][colOffsetB] = B[(loadInd + rowOffsetB + blockOffset)*N + colOffsetB + blockIdx.x*BN];
    }
    __syncthreads();

    for (int dotIdx=0; dotIdx<BK; ++dotIdx){
      for (int i = 0; i < TM; ++i){
        regM[i] = As[i + threadRow*TM][dotIdx];
      }
      for (int i = 0; i<TN; ++i){
        regN[i] = Bs[dotIdx][i + TN*threadCol];
      }

      for (int resIdM = 0; resIdM<TM; ++resIdM){
        for (int resIdN=0; resIdN<TN; ++resIdN){
          threadResults[resIdM][resIdN] += regM[resIdM] * regN[resIdN];

        }
      }
    }
    __syncthreads();
  }

  for (int resIdM = 0; resIdM<TM; ++resIdM){
    for (int resIdN=0; resIdN<TN; ++resIdN){
      int c_ind = (resIdM + threadRow*TM + blockIdx.y*BM)*N + threadCol*TN + BN*blockIdx.x + resIdN;
      C[c_ind]= alpha * threadResults[resIdM][resIdN] + beta * C[c_ind];
    }
  }

}