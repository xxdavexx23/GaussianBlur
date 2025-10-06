#pragma once
#include <string>
#include <opencv2/core.hpp>


namespace image_helpers {

    const int DEFAULT_KSIZE = 3;
    
    // ---- Validation ----
    bool isOdd(int v);
    bool validateKernelSize(int k);  // true if odd and >= 3
    bool validateSigma(float s);     // true if > 0
    float deriveSigmaFromKsize(int ksize);

    // ---- I/O helpers ----
    // Load an image as grayscale (CV_8UC1), throw or exit on error
    cv::Mat loadGrayOrDie(const std::string& path);

    // Load an image as BGR color (CV_8UC3)
    cv::Mat loadBGROrDie(const std::string& path);

    // Save an image, throw or exit on error
    void saveOrDie(const std::string& path, const cv::Mat& img);

} // namespace image_helpers
