#pragma once
#include <opencv2/core.hpp>

namespace gauss
{

  // Border handling for sampling outside image bounds.
  enum class Border
  {
    Replicate, // clamp to nearest edge pixel
    Reflect,   // ...c b a | a b c | b a...
    Constant   // pad with a constant value (0 for v1)
  };

  // Tunable parameters for Gaussian blur.
  struct BlurParams
  {
    int ksize = 7;      // must be odd and >= 3
    float sigma = 1.2f; // if <= 0, derive from ksize
    Border border = Border::Replicate;
  };

  // ---- Public API ----
  // CPU Gaussian blur on a single-channel 8-bit image (grayscale).
  // - src: CV_8UC1
  // - dst: CV_8UC1 (created/resized inside the function)
  // - params: kernel size, sigma, border rule (see above)
  void gaussianBlurCPU(const cv::Mat &src, cv::Mat &dst, const BlurParams &params);

  void gaussianBlurCPUOptimized(const cv::Mat &src, cv::Mat &dst, const BlurParams &params);
  // Optional convenience: apply to 3-channel BGR (per-channel blur).
  // You can implement this later; keeping the declaration here lets callers use it.
  void gaussianBlurCPU_BGR(const cv::Mat &srcBGR, cv::Mat &dstBGR, const BlurParams &params);

  // Also a host function (no __global__), implemented in a .cu file
  void gaussianBlurCUDA(const cv::Mat &src, cv::Mat &dst, const BlurParams &params);

  // ---- Helpers (declarations only; implement in blur.cpp) ----

  // Validate and normalize parameters:
  // - Ensures ksize is odd and >= 3 (adjust or throw, your choice in impl).
  // - If sigma <= 0, derive from ksize via a standard heuristic.
  // - Returns a sanitized copy of params.
  BlurParams normalizeParams(const BlurParams &params);

  // Build kernels (useful for testing/separable version later).
  // 1D kernel sums to 1.0 (type CV_32F, size ksize x 1).
  cv::Mat makeGaussianKernel1D(int ksize, float sigma);

  // 2D kernel sums to 1.0 (type CV_32F, size ksize x ksize).
  cv::Mat makeGaussianKernel2D(int ksize, float sigma);

} // namespace gauss
