#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  int blockHeight = BM/TM;
  int blockWidth = BN/TN;

  __shared__ float As[BK][BM];
  __shared__ float Bs[BK*BN];

  int threadRow = threadIdx.x / blockWidth;
  int threadCol = threadIdx.x % blockWidth;

  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  float threadResults[TM][TN] = {0.0};

  int strideA = blockDim.x / BK;
  int strideB = blockDim.x / BN;

  int rowOffsetA = threadIdx.x / (BK/4);
  int colOffsetA = threadIdx.x % (BK/4);

  int rowOffsetB = threadIdx.x / (BN/4);
  int colOffsetB = threadIdx.x % (BN/4);

  // Chunks of size BMxBK and BKxBN
  for(int blockOffset = 0; blockOffset<K; blockOffset += BK){

    float4 tmp =
    reinterpret_cast<float4 *>(&A[(rowOffsetA+blockIdx.y*BM) * K + blockOffset + colOffsetA*4])[0];
    // transpose A during the GMEM to SMEM transfer
    As[(colOffsetA*4 + 0)][rowOffsetA] = tmp.x;
    As[(colOffsetA*4 + 1)][rowOffsetA] = tmp.y;
    As[(colOffsetA*4 + 2)][rowOffsetA] = tmp.z;
    As[(colOffsetA*4 + 3)][rowOffsetA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[rowOffsetB*BN + colOffsetB * 4])[0] =
        reinterpret_cast<float4 *>(&B[(rowOffsetB + blockOffset) * N + blockIdx.x*BN + colOffsetB * 4])[0];

    __syncthreads();

    for (int dotIdx=0; dotIdx<BK; ++dotIdx){
      for (int i = 0; i < TM; ++i){
        regM[i] = As[dotIdx][i + threadRow*TM];
      }
      for (int i = 0; i<TN; ++i){
        regN[i] = Bs[dotIdx*BN + i + TN*threadCol];
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