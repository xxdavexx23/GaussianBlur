#include "gauss/blur.hpp"
#include "gauss/image.hpp"
#include <cmath>
#include <iostream>
using gauss::BlurParams;
using gauss::Border;
using image_helpers::DEFAULT_KSIZE;

BlurParams gauss::normalizeParams(const BlurParams &params)
{
    int ksize = params.ksize;
    float sigma = params.sigma;
    Border border = params.border;

    if (!image_helpers::validateKernelSize(ksize))
    {
        if (ksize < 3)
        {
            ksize = DEFAULT_KSIZE;
        }
        else if (!image_helpers::isOdd(ksize))
        {
            ksize = ksize + 1;
        }
        std::cerr << "Warning: Ksize not valid, adjusting ksize to " << ksize << std::endl;
    }

    if (sigma <= 0.0f)
    {
        sigma = image_helpers::deriveSigmaFromKsize(ksize);
        std::cerr << "Warning: derived sigma -  " << sigma << std::endl;
    }

    return {ksize, sigma, border};
}

cv::Mat gauss::makeGaussianKernel1D(int ksize, float sigma)
{
    cv::Mat kernel(1, ksize, CV_32F);
    int radius = (ksize - 1) / 2;
    float sum = 0.0f;

    for (int i = 0; i < ksize; i++)
    {
        float d = i - radius; // column offset
        float val = std::exp(-((d * d) / (2.0f * sigma * sigma)));
        kernel.at<float>(0, i) = val;
        sum += val;
    }

        // normalize
    for (int i = 0; i < ksize; i++)
    {
        kernel.at<float>(0, i) /= sum;
    }

    return kernel;

}

cv::Mat gauss::makeGaussianKernel2D(int ksize, float sigma)
{
    cv::Mat kernel(ksize, ksize, CV_32F);
    int radius = (ksize - 1) / 2;
    float sum = 0.0f;

    // build raw weights
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            float dx = j - radius; // column offset
            float dy = i - radius; // row offset
            float val = std::exp(-((dx * dx + dy * dy) / (2.0f * sigma * sigma)));
            kernel.at<float>(i, j) = val;
            sum += val;
        }
    }

    // normalize
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            kernel.at<float>(i, j) /= sum;
        }
    }

    return kernel;
}

void gauss::gaussianBlurCPU(const cv::Mat& src, cv::Mat& dst, const BlurParams& params)
{

    if(src.empty())
    {
        std::cerr << "Error: Image to blur does not exist " << std::endl;
        exit(1);
    }

    if(src.type() != CV_8UC1)
    {
        std::cerr << "Error: Image is not in gray scale - Load Gray Scale instead " << std::endl;
        exit(1);
    }


    BlurParams norm = normalizeParams(params);

    int k = norm.ksize;
    float sigma = norm.sigma;
    int radius = (k-1) / 2;

    cv::Mat kernel =  makeGaussianKernel2D(k, sigma);
    
    cv::Size s = src.size();
    int width = s.width;
    int height = s.height;

    dst.create(src.size(), CV_8UC1);
    
    for(int r = 0; r < height; r++ )
    {
        for(int c = 0; c < width; c++)
        {
            float accum = 0.0f;

            for(int dy = -radius; dy <= radius; dy++)
            {
                for(int dx = -radius; dx <= radius; dx++)
                {
                    int rr = (r + dy);
                    int cc = (c + dx);

                    if (rr < 0)
                    {
                        rr = 0;
                    }
                    else if(rr >= height)
                    {
                        rr = height - 1;
                    }
                    
                    if(cc < 0)
                    {
                        cc = 0;
                    }
                    else if(cc >= width)
                    {
                        cc = width - 1;
                    }

                    float pixel_weight = static_cast<float>(src.at<uchar>(rr,cc));
                    float kernel_weight = kernel.at<float>(dy + radius, dx + radius);

                    accum += pixel_weight * kernel_weight;

                }
            }

            accum = std::round(accum);

            if (accum > 255)
            {
                accum = 255;
            }
            else if (accum < 0)
            {
                accum = 0;
            }

            dst.at<uchar>(r,c) = static_cast<uchar>(accum);
            
        }
    }
    
}

// --- Separable Gaussian blur: src (CV_8UC1) -> dst (CV_8UC1) ---
void gauss::gaussianBlurCPUOptimized(const cv::Mat& src, cv::Mat& dst, const BlurParams& params)
{
    if (src.empty()) {
        std::cerr << "Error: Image to blur does not exist.\n";
        std::exit(1);
    }
    if (src.type() != CV_8UC1) {
        std::cerr << "Error: Expected CV_8UC1 (grayscale). Load grayscale first.\n";
        std::exit(1);
    }

    const BlurParams p = normalizeParams(params);
    const int k = p.ksize;
    const float sigma = p.sigma;
    const int r = (k - 1) / 2;

    // Build 1D kernel once
    cv::Mat k1 = makeGaussianKernel1D(k, sigma);

    const int h = src.rows;
    const int w = src.cols;

    // Pass 1: horizontal (src -> tmp), keep float precision between passes
    cv::Mat tmp(h, w, CV_32F);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int d = -r; d <= r; ++d) {
                int xx = x + d;
                // replicate clamp on X
                if (xx < 0) xx = 0;
                else if (xx >= w) xx = w - 1;

                float pix = static_cast<float>(src.at<uchar>(y, xx));
                float wgt = k1.at<float>(0, d + r);
                acc += wgt * pix;
            }
            tmp.at<float>(y, x) = acc; // no rounding/clamping here
        }
    }

    // Pass 2: vertical (tmp -> dst)
    dst.create(h, w, CV_8UC1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int d = -r; d <= r; ++d) {
                int yy = y + d;
                // replicate clamp on Y
                if (yy < 0) yy = 0;
                else if (yy >= h) yy = h - 1;

                float pix = tmp.at<float>(yy, x);
                float wgt = k1.at<float>(0, d + r);
                acc += wgt * pix;
            }
            // round + clamp, write uchar
            acc = std::round(acc);
            if (acc < 0.f) acc = 0.f;
            else if (acc > 255.f) acc = 255.f;

            dst.at<uchar>(y, x) = static_cast<uchar>(acc);
        }
    }
}