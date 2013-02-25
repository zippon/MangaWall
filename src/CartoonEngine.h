//
//  CartoonEngine.h
//  Im2Cartoon
//
//  Created by Zhipeng Wu on 2/15/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#ifndef __Im2Cartoon__CartoonEngine__
#define __Im2Cartoon__CartoonEngine__

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
using std::cout;
using std::endl;
using std::string;

class CartoonEngine {
public:
  // Constructors:
  explicit CartoonEngine(const string& file_path) {
    image_ = cv::imread(file_path, 1);
    if (!image_.empty()) {
      rows_ = image_.rows;
      cols_ = image_.cols;
    }
  }
  explicit CartoonEngine(const cv::Mat& image) {
    if (!image.empty()) {
      if (3 == image.channels()) {
        image.copyTo(image_);
      } else {
        cv::cvtColor(image, image_, CV_GRAY2BGR);
      }
      rows_ = image_.rows;
      cols_ = image_.cols;
    }
  }
  
  // Accessers:
  const cv::Mat& cartoon() const {
    return cartoon_;
  }
  
  // Manga conversion:
  bool Convert2Cartoon(int iter_num,
                       int d,
                       double sigma_color,
                       double sigma_space);
  bool Convert2Cartoon() {
    return Convert2Cartoon(10, 5, 30, 400);
  }
  
private:
  cv::Mat image_;      // Original input image.
  cv::Mat cartoon_;    // Converted cartoon image.
  int rows_;           // Original image row number.
  int cols_;           // Original image col number.
  
  // Disallow copy and assign.
  void operator= (const CartoonEngine&);
  CartoonEngine(const CartoonEngine&);
};

#endif /* defined(__Im2Cartoon__CartoonEngine__) */
