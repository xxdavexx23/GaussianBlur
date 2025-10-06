#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include "gauss/blur.hpp"

// ---- Config ----
#ifndef MAX_K
#define MAX_K 129 // supports ksize up to 129; bump if you need more
#endif

__constant__ float d_k1[MAX_K];

__device__ __forceinline__ int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void horizontalPassKernel(
    const unsigned char *__restrict__ src, int width, int height, size_t srcPitchBytes,
    float *__restrict__ tmp, size_t tmpPitchBytes,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // row pointers using pitch (bytes)
    const unsigned char *srcRow = (const unsigned char *)((const char *)src + y * srcPitchBytes);
    float acc = 0.0f;

    // 1D convolution along X with replicate borders
    for (int d = -radius; d <= radius; ++d)
    {
        int xx = clamp_int(x + d, 0, width - 1);
        float pix = static_cast<float>(srcRow[xx]);
        float wgt = d_k1[d + radius]; // d_k1 in __constant__ memory
        acc += wgt * pix;
    }

    float *tmpRow = (float *)((char *)tmp + y * tmpPitchBytes);
    tmpRow[x] = acc; // keep as float; rounding happens after vertical pass
}

// Vertical pass: tmp(float) -> dst(uchar)
__global__ void verticalPassKernel(
    const float *__restrict__ tmp, int width, int height, size_t tmpPitchBytes,
    unsigned char *__restrict__ dst, size_t dstPitchBytes,
    int radius)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float acc = 0.f;

// convolution along Y
#pragma unroll
    for (int d = -radius; d <= radius; ++d)
    {
        int yy = clamp_int(y + d, 0, height - 1);
        const float *tmpRow = (const float *)((const char *)tmp + yy * tmpPitchBytes);
        float pix = tmpRow[x];
        float w = d_k1[d + radius];
        acc += w * pix;
    }

    // round + clamp to 8-bit
    acc = roundf(acc);
    if (acc < 0.f)
        acc = 0.f;
    if (acc > 255.f)
        acc = 255.f;

    unsigned char *dstRow = (unsigned char *)((char *)dst + y * dstPitchBytes);
    dstRow[x] = static_cast<unsigned char>(acc);
}

namespace gauss
{

    static inline void cudaCheck(cudaError_t e, const char *msg)
    {
        if (e != cudaSuccess)
        {
            std::cerr << "[CUDA] " << msg << " : " << cudaGetErrorString(e) << std::endl;
            std::exit(1);
        }
    }
    __host__ void gaussianBlurCUDA(const cv::Mat &src, cv::Mat &dst, const BlurParams &params)
    {

        if (src.empty())
        {
            std::cerr << "gaussianBlurCUDA: empty src\n";
            std::exit(1);
        }
        if (src.type() != CV_8UC1)
        {
            std::cerr << "gaussianBlurCUDA: expected CV_8UC1 grayscale\n";
            std::exit(1);
        }

        // Normalize parameters (ksize odd>=3, derive sigma if needed)
        BlurParams p = normalizeParams(params);
        const int ksize = p.ksize;
        const int radius = (ksize - 1) / 2;

        if (ksize > MAX_K)
        {
            std::cerr << "gaussianBlurCUDA: ksize (" << ksize << ") exceeds MAX_K (" << MAX_K << ")\n";
            std::exit(1);
        }

        // Build 1D kernel on host (row vector CV_32F, normalized)
        cv::Mat k1 = makeGaussianKernel1D(ksize, p.sigma);

        // Copy kernel to constant memory
        cudaCheck(cudaMemcpyToSymbol(d_k1, k1.ptr<float>(0), ksize * sizeof(float)),
                  "MemcpyToSymbol(d_k1)");

        const int width = src.cols;
        const int height = src.rows;

        // Device buffers (pitched)
        unsigned char *d_src = nullptr;
        size_t srcPitch = 0;
        float *d_tmp = nullptr;
        size_t tmpPitch = 0;
        unsigned char *d_dst = nullptr;
        size_t dstPitch = 0;

        // Allocate pitched memory
        cudaCheck(cudaMallocPitch((void **)&d_src, &srcPitch, width * sizeof(unsigned char), height),
                  "cudaMallocPitch d_src");
        cudaCheck(cudaMallocPitch((void **)&d_tmp, &tmpPitch, width * sizeof(float), height),
                  "cudaMallocPitch d_tmp");
        cudaCheck(cudaMallocPitch((void **)&d_dst, &dstPitch, width * sizeof(unsigned char), height),
                  "cudaMallocPitch d_dst");

        // Copy src -> d_src (2D because of pitch)
        cudaCheck(cudaMemcpy2D(d_src, srcPitch,
                               src.data, src.step,
                               width * sizeof(unsigned char), height,
                               cudaMemcpyHostToDevice),
                  "Memcpy2D H2D src");

        // Kernel launch config
        dim3 block(32, 8);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        // Horizontal pass
        horizontalPassKernel<<<grid, block>>>(
            d_src, width, height, srcPitch,
            d_tmp, tmpPitch,
            radius);

        cudaCheck(cudaGetLastError(), "horizontalPassKernel launch");
        cudaCheck(cudaDeviceSynchronize(), "horizontalPassKernel sync");

        // Vertical pass
        verticalPassKernel<<<grid, block>>>(
            d_tmp, width, height, tmpPitch,
            d_dst, dstPitch,
            radius);

        cudaCheck(cudaGetLastError(), "verticalPassKernel launch");
        cudaCheck(cudaDeviceSynchronize(), "verticalPassKernel sync");

        // Prepare dst and copy back
        dst.create(height, width, CV_8UC1);
        cudaCheck(cudaMemcpy2D(dst.data, dst.step,
                               d_dst, dstPitch,
                               width * sizeof(unsigned char), height,
                               cudaMemcpyDeviceToHost),
                  "Memcpy2D D2H dst");

        // Cleanup
        cudaFree(d_src);
        cudaFree(d_tmp);
        cudaFree(d_dst);
    }

}