#include "gauss/image.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>  

bool image_helpers::isOdd(int v) {
    return (v % 2 != 0);
}

bool image_helpers::validateKernelSize(int k)
{
    return (k >= 3) && isOdd(k);
}

bool image_helpers::validateSigma(float s)
{
    return (s > 0.0f);
}

float image_helpers::deriveSigmaFromKsize(int kSize)
{
    if (!validateKernelSize(kSize))
    {
        return 0.0f;
    }

    else
    {  
        return 0.3f * ((static_cast<float>(kSize - 1) / 2.0f) - 1.0f) + 0.8f;
    }
}


cv::Mat image_helpers::loadGrayOrDie(const std::string& path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr<< "Error: could not read image from image path - "<< path << std::endl;
        exit(1);
    }

    if(image.type() != CV_8UC1)
    {
        std::cerr<< "Error: image type wrong " << std::endl;
        exit(1);
    }

    return image;

    
}

cv::Mat image_helpers::loadBGROrDie(const std::string& path)
{
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr<< "Error: could not read image from image path - "<< path << std::endl;
        exit(1);
    }

    if(image.type() != CV_8UC3)
    {
        std::cerr<< "Error: image type wrong " << std::endl;
        exit(1);
    }

    return image;
}


void image_helpers::saveOrDie(const std::string& path, const cv::Mat& img)
{

    if(!cv::imwrite(path, img))
    {
        std::cerr <<"Error: cannot save image to location - "<< path << std::endl;
        exit(1);
    }
    

}