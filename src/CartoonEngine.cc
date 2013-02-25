//
//  CartoonEngine.cpp
//  Im2Cartoon
//
//  Created by Zhipeng Wu on 2/15/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#include "CartoonEngine.h"

bool CartoonEngine::Convert2Cartoon(int iter_num,
                                    int d,
                                    double sigma_color,
                                    double sigma_space) {
  if ((image_.empty()) || (CV_8UC3 != image_.type()) || (iter_num < 1))
    return false;
  cv::Mat cartoon_img = image_;
  cv::Mat temp;
  // Step 1: resize the original image to save pprocessing time (if needed).
  cv::Size2i small_size(cols_, rows_);
  if ((rows_ > 500) || (cols_ > 500)) {
    if (rows_ > cols_) {
      small_size.height = 500;
      small_size.width =
      static_cast<int>(500 * (static_cast<float>(cols_) / rows_));
    } else {
      small_size.width = 500;
      small_size.height =
      static_cast<int>(500 * (static_cast<float>(rows_) / cols_));
    }
    cv::resize(cartoon_img, cartoon_img, small_size, cv::INTER_LINEAR);
    temp.create(small_size, CV_8UC3);
  } else {
    temp.create(rows_, cols_, CV_8UC3);
  }
  // Step 2: do bilateral filtering for several times for cartoon rendition.
  for (int i = 0; i < iter_num; ++i) {
    cv::bilateralFilter(cartoon_img, temp, d, sigma_color, sigma_space);
    cv::bilateralFilter(temp, cartoon_img, d, sigma_color, sigma_space);
  }
  // Step 3: revert the original image size.
  if (small_size.width != cols_) {
    cv::resize(cartoon_img, cartoon_img, cv::Size2i(cols_, rows_));
  }
  cartoon_img.copyTo(cartoon_);
  return true;
}