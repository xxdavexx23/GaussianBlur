# Gaussian Blur (CPU & CUDA)

This project demonstrates how to implement **Gaussian Blur** on images using both:
- **Sequential CPU processing** (naive and separable versions), and  
- **Parallel CUDA GPU processing** (two-pass separable convolution with constant memory).

The goal is to show the step-by-step transition from a simple CPU-based algorithm to an efficient GPU implementation.

---

## 📖 Overview

Gaussian Blur is a widely used image processing operation that smooths images and reduces noise by applying a Gaussian kernel.  

This repo implements Gaussian Blur in three flavors:

1. **CPU Naive (2D convolution)**  
   - Direct convolution with a full 2D Gaussian kernel.  
   - Simple to understand but computationally expensive for large kernels.

2. **CPU Optimized (Separable convolution)**  
   - Uses the separability of Gaussian kernels to apply two 1D convolutions (horizontal + vertical).  
   - Reduces complexity from O(k²) → O(2k).

3. **CUDA GPU (Separable convolution)**  
   - Parallelizes the horizontal and vertical passes across thousands of GPU threads.  
   - Uses constant memory for the Gaussian kernel.  
   - Supports border handling (replicate for v1).  

---

## 🗂 Project Structure

```
.
├── CMakeLists.txt           # Build configuration
├── include/gauss/           # Public headers
│   ├── blur.hpp             # Blur APIs (CPU + CUDA)
│   └── image.hpp            # Image I/O & helpers
├── src/
│   ├── main.cpp             # Entry point (loads, blurs, saves)
│   ├── blur.cpp             # CPU blur implementations
│   ├── utils.cpp            # Utility functions (kernel validation, sigma derivation)
│   └── gauss_cuda.cu        # CUDA kernels + host wrapper
└── data/
    ├── photo-...jpg         # Sample input image
    └── results/             # Output images
```

---

## ⚙️ Build Instructions

### Prerequisites
- CMake ≥ 3.16
- C++17 compiler
- OpenCV (core, imgproc, imgcodecs modules)
- CUDA Toolkit (for GPU build)

### Configure and build

**CPU-only build:**
```bash
cmake -S . -B build -DGAUSS_USE_CUDA=OFF
cmake --build build
```

**With CUDA (e.g., GTX 1080 Ti, SM 61):**
```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build
```

---

## ▶️ Usage

Run the executable:

```bash
./build/gauss_cpu
```

Default pipeline in `main.cpp`:
1. Load an image (grayscale).  
2. Save the original to `data/results/`.  
3. Apply Gaussian Blur with chosen kernel size and sigma.  
4. Save blurred output.

Modify `main.cpp` to select between CPU and CUDA versions:

```cpp
gauss::gaussianBlurCPU(image, blur_image, params);
// or
gauss::gaussianBlurCUDA(image, blur_image, params);
```

### Example output
- `cat-photo.jpg` → input grayscale image  
- `cat-photo_blur.jpg` → blurred result

---

## 📊 Performance Notes

- **CPU naive 2D**: Very slow for large kernels (O(k²) per pixel).  
- **CPU separable**: ~10× faster than naive for larger kernels (O(2k)).  
- **CUDA separable**: Orders of magnitude faster on large images, especially with bigger kernels (e.g., k=31, 61, 101).  

You can test by varying:
```cpp
gauss::BlurParams params = {31, 0};  // 31×31 kernel, auto sigma
```

---

## 🚀 Next Steps

- [ ] Add shared-memory tiling for further CUDA speedup.  
- [ ] Implement other border modes (Reflect, Constant).  
- [ ] Add 3-channel (BGR) support.  
- [ ] Benchmark systematically and add charts.  
- [ ] Write unit tests comparing CPU vs CUDA outputs (mean absolute difference).  

---

## 📚 References

- [OpenCV Documentation: GaussianBlur](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)  
- NVIDIA CUDA Programming Guide  
- Gonzalez & Woods, *Digital Image Processing*  
