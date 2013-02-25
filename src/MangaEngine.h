//
//  MangaEngine.h : Creating Manga-like image with single input photo.
//  Im2Manga
//
//  Created by Zhipeng Wu on 1/28/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#ifndef __Im2Manga__MangaEngine__
#define __Im2Manga__MangaEngine__

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
using std::cout;
using std::endl;
using std::string;

class MangaEngine {
public:
  // Constructors:
  explicit MangaEngine(const string& file_path) {
    image_ = cv::imread(file_path, 0);
    if (!image_.empty()) {
      rows_ = image_.rows;
      cols_ = image_.cols;
    }
  }
  explicit MangaEngine(const cv::Mat& image) {
    if (!image.empty()) {
      if (1 == image.channels()) {
        image.copyTo(image_);
      } else {
        cv::cvtColor(image, image_, CV_RGB2GRAY);
      }
      rows_ = image_.rows;
      cols_ = image_.cols;
    }
  }
  
  // Accessers:
  const cv::Mat& texture() const {
    return texture_;
  }
  const cv::Mat& structure() const {
    return structure_;
  }
  const cv::Mat& manga() const {
    return manga_;
  }
  
  // Manga conversion:
  bool Convert2Manga(float sigma, float thresh1, float thresh2, float theta) {
    if (!ExtractStructure(sigma, thresh1, thresh2))
      return false;
    if (!ExtractTexture(theta))
      return false;
    manga_ = structure_.mul(texture_);
    return true;
  }
  bool Convert2Manga() {
    return Convert2Manga(1.0, 100, 1.0, 0.7);
  }
  
  // Functions:
  // Adding frame template for generated manga.
  bool AddFrameTemplate(const string& frame_path);
  bool AddText(const string& text, const string& dialog_path, const cv::Point2f left_top);
  
private:
  bool ExtractStructure(float sigma, float thresh1, float thresh2);
  // Functions used by ExtractStructure:
  void GetGaussianWeights(float* weights, int neighbor, float sigma);
  void GetDiffGaussianWeights(float* weights, int neighbor, float sigma);
  void GetDevGaussianWeights(float* weights, int neighbor, float sigma);
  
  bool ExtractTexture(float theta);
  // Functions used by ExtractTexture:
  bool ToneMapping(cv::Mat* tone_mapping);
  bool Halftoning(cv::Mat* halftoning);
  bool HistSpecification(const cv::Mat& h_target, cv::Mat* tone_mapping);
  
  cv::Mat image_;      // Original input image (single-channel/gray-scale image)
  // type: CV_8UC1 [0, 255]
  cv::Mat texture_;    // Texture rendering result.
  // type: CV_32FC1 [0, 1]
  cv::Mat structure_;  // Structure extraction result.
  // type: CV_32FC1 [0, 1]
  cv::Mat manga_;      // Manga image as a combination of texture_ and structure_.
  int rows_;           // Original image row number.
  int cols_;           // Original image col number.
  
  // Disallow copy and assign.
  void operator= (const MangaEngine&);
  MangaEngine(const MangaEngine&);
};

#endif /* defined(__Im2Manga__MangaEngine__) */
