#include<iostream>
#include "gauss/image.hpp"
#include "gauss/blur.hpp"

int main()
{
    const std::string image_path= "data/photo-1529778873920-4da4926a72c2.jpg";
    const std::string image_dest= "data/results/cat-photo.jpg";
    cv::Mat image = image_helpers::loadGrayOrDie(image_path);
    image_helpers::saveOrDie(image_dest, image);


    const std::string image_dest_blur = "data/results/cat-photo_blur.jpg";

    cv::Mat blur_image;
    gauss::BlurParams params = {100, 0};

    gauss::gaussianBlurCUDA(image, blur_image, params);

    image_helpers::saveOrDie(image_dest_blur, blur_image);
    return 0;
}