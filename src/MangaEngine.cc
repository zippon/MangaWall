//
//  MangaEngine.cpp
//  Im2Manga
//
//  Created by Zhipeng Wu on 1/28/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#include "MangaEngine.h"
#include <math.h>

const float PI = 3.1415926;

// Given an inuput image (image_), extract the edges to obtain the structure
// component of manga image.
bool MangaEngine::ExtractStructure(float sigma, float thresh1, float thresh2) {
  if (image_.empty())
    return false;
  
  // structure_ is a CV_32FC1 image ranging in [0, 1], where 1 stands for edges.
  if (!structure_.empty())
    structure_.release();
  structure_ = cv::Mat::ones(rows_, cols_, CV_32FC1);
  
  int neighbor = round(sigma * 3);
  // If neighbor = k, we sample 2k + 1 pixels along the gradient direction and
  // centered at the current pixel. Following are the weighting vectors for
  // sampled pixels.
  // 1. weights for gaussian first derivative.
  float* dev_gaussian_weights = new float[neighbor * 2 + 1];
  // 2. weights for difference of gaussian.
  float* diff_gaussian_weights = new float[neighbor * 2 + 1];
  GetDevGaussianWeights(dev_gaussian_weights, neighbor, sigma);
  GetDiffGaussianWeights(diff_gaussian_weights, neighbor, sigma);
  
  // Get sobel gradients.
  cv::Mat gx, gy;
  cv::Sobel(image_, gx, CV_32FC1, 1, 0);
  cv::Sobel(image_, gy, CV_32FC1, 0, 1);
  
  
  for (int r = neighbor; r < (rows_ - neighbor); ++r) {
    for (int c = neighbor; c < (cols_ - neighbor); ++c) {
      // Step 1: Get pixel gradient direction.
      float dx = gx.at<float>(r, c);
      float dy = gy.at<float>(r, c);
      float dr = sqrt(pow(dx, 2) + pow(dy, 2));
      if (dr < thresh1) {
        // Current pixels is not a candidate edge.
        continue;
      } else {
        float sin_theta = dy / dr;
        float cos_theta = dx / dr;
        // Step 2: Sample neighbor pixels along the gradient direction with gaussian
        // weighting and dog weighting.
        float* sample_pixels = new float[neighbor * 2 + 1];
        sample_pixels[neighbor] = static_cast<float>(image_.at<uchar>(r, c));
        for (int k = 1; k <= neighbor; ++k) {
          int r_offset = round(sin_theta * k);
          int c_offset = round(cos_theta * k);
          sample_pixels[neighbor + k] =
          static_cast<float>(image_.at<uchar>(r + r_offset, c + c_offset));
          sample_pixels[neighbor - k] =
          static_cast<float>(image_.at<uchar>(r - r_offset, c - c_offset));
        }
        // Step 3: Calculate edge response and thresholding with 'thresh'.
        float sum_diff = 0;
        float sum_dev = 0;
        for (int k = 0; k < 2 * neighbor + 1; ++k) {
          sum_diff += sample_pixels[k] * diff_gaussian_weights[k];
          sum_dev += sample_pixels[k] * dev_gaussian_weights[k];
        }
        float response = fabs(sum_dev) - fabs(sum_diff);
        if (response < thresh2)
          // 1.0 means the current pixel is a white/non-edge pixel.
          structure_.at<float>(r, c) = 1.0;
        else {
          // 0.0 means the current pixel is a black/edge pixel.
          structure_.at<float>(r, c) = 0.0;
        }
        delete[] sample_pixels;
      }
    }
  }
  delete[] dev_gaussian_weights;
  delete[] diff_gaussian_weights;
  return true;
}

// Prepare 1-d gaussian template.
void MangaEngine::GetGaussianWeights(float* weights,
                                     int neighbor,
                                     float sigma) {
  if ((NULL == weights) || (neighbor < 0))
    return;
  float term1 = 1.0 / (sqrt(2.0 * PI) * sigma);
  float term2 = -1.0 / (2 * pow(sigma, 2));
  weights[neighbor] = term1;
  for (int i = 1; i <= neighbor; ++ i) {
    weights[neighbor + i] = exp(pow(i, 2) * term2) * term1;
    weights[neighbor - i] =  weights[neighbor + i];
  }
}

// Prepare 1-d gaussian derivative template.
void MangaEngine::GetDevGaussianWeights(float* weights,
                                        int neighbor,
                                        float sigma) {
  if ((NULL == weights) || (neighbor < 0))
    return;
  float term1 = -1.0 / (sqrt(2 * PI) * pow(sigma, 3));
  float term2 = -1.0 / (2 * pow(sigma, 2));
  weights[neighbor] = 0;
  for (int i = 1; i <= neighbor; ++ i) {
    weights[neighbor + i] = exp(pow(i, 2) * term2) * term1 * i;
    weights[neighbor - i] = -1 * weights[neighbor + i];
  }
}

// Prepare 1-d difference of gaussian template.
void MangaEngine::GetDiffGaussianWeights(float* weights,
                                         int neighbor,
                                         float sigma) {
  if ((NULL == weights) || (neighbor < 0))
    return;
  float* gaussian_c = new float[neighbor * 2 + 1];
  float* gaussian_s = new float[neighbor * 2 + 1];
  GetGaussianWeights(gaussian_c, neighbor, sigma);
  GetGaussianWeights(gaussian_s, neighbor, sigma * 1.6);
  for (int i = 0; i < neighbor * 2 + 1; ++i) {
    weights[i] = gaussian_c[i] - gaussian_s[i];
  }
  delete[] gaussian_c;
  delete[] gaussian_s;
}

// Given an inuput image (image_), render the manga-like texture.
bool MangaEngine::ExtractTexture(float theta) {
  if (image_.empty())
    return false;
  // texture_ is a CV_32FC1 image ranging in [0, 1].
  if (!texture_.empty())
    texture_.release();
  texture_ = cv::Mat::zeros(rows_, cols_, CV_32FC1);
  
  cv::Mat tone_mapping;
  cv::Mat halftoning;
  
  // Step 1: tone mapping. [0, 1], CV32FC1
  if (!ToneMapping(&tone_mapping))
    return false;
  // Step 2: halftoning by using bayer 2x2 templates. [0, 1],  CV32FC1
  if (!Halftoning(&halftoning))
    return false;
  
  // Step 3: combining tone_mapping and haltoning into the texture rendering result.
  texture_ = theta * tone_mapping + (1 - theta) * halftoning;
  return true;
}

// Tone mapping with pre-defined histogram.
bool MangaEngine::ToneMapping(cv::Mat* tone_mapping) {
  if (NULL == tone_mapping)
    return false;
  
  // Prepare tartget cumulative hisotgram.
  cv::Mat c_target(256, 1, CV_32FC1);
  cv::Mat h_bright(256, 1, CV_32FC1);
  cv::Mat h_dark(256, 1, CV_32FC1);
  cv::Mat h_mid(256, 1, CV_32FC1);
  float sum_bright = 0;
  float sum_dark = 0;
  float sum_mid = 0;
  float sum = 0;
  for (int i = 0; i < 256; ++i) {
    // bright tone.
    h_bright.at<float>(i) = exp((i - 255) / 9);
    sum_bright += h_bright.at<float>(i);
    // dark tone.
    h_dark.at<float>(i) = exp(i / -9);
    sum_dark += h_dark.at<float>(i);
    // middle tone
    if ((i <= 225) && (i >= 105))
      h_mid.at<float>(i) = 1;
    else
      h_mid.at<float>(i) = 0;
    sum_mid += h_mid.at<float>(i);
  }
  
  float bright, dark, mid;
  for (int i = 0; i < 256; ++i) {
    bright = h_bright.at<float>(i) / sum_bright;
    dark = h_dark.at<float>(i) / sum_dark;
    mid = h_mid.at<float>(i) / sum_mid;
    sum += (10 * bright + mid + 5 * dark) / 16;
    c_target.at<float>(i) = sum;
  }
  
  // Tone mapping by using histogram specification.
  if(!HistSpecification(c_target, tone_mapping))
    return false;
  else
    return true;
}

// Ordered dithering by using Bayer template (2x2).
bool MangaEngine::Halftoning(cv::Mat* halftoning) {
  if (NULL == halftoning)
    return false;
  halftoning->create(rows_, cols_, CV_32FC1);
  uchar bayer[4] = {102, 153, 204, 51};
  for (int r = 0; r < rows_ ; ++r) {
    for (int c = 0; c < cols_ ; ++c) {
      int ind = r % 2 * 2 + c % 2;
      if (image_.at<uchar>(r, c) >= bayer[ind]) {
        halftoning->at<float>(r, c) = 1.0;
      } else {
        halftoning->at<float>(r, c) = 0.0;
      }
    }
  }
  return true;
}

// Histogram specification by using group mapping.
bool MangaEngine::HistSpecification(const cv::Mat& c_target, cv::Mat* tone_mapping) {
  if (c_target.empty() || image_.empty())
    return false;
  // The mapping of pixel values from original image to target image.
  cv::Mat hist_map(1, 256, CV_8UC1);
  // Histogram for original image. (CV_32FC1)
  cv::Mat h_ori;
  float range[] = {0, 255};
  const float* ranges = {range};
  int histSize = 256;
  cv::calcHist(&image_, 1, 0, cv::Mat(), h_ori, 1, &histSize, &ranges, true, false);
  // Cumulative histogram for original image.
  cv::Mat c_ori(256, 1, CV_32FC1);
  float sum = 0;
  for (int i = 0; i < 256; ++i) {
    sum += h_ori.at<float>(i);
    c_ori.at<float>(i) = sum;
  }
  for (int i = 0; i < 256; ++i) {
    c_ori.at<float>(i) /= sum;
  }
  
  // Distance matrix.
  cv::Mat dist(256, 256, CV_32FC1);
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      dist.at<float>(x, y) = fabs(c_ori.at<float>(y) - c_target.at<float>(x));
    }
  }
  
  // Construct mapping index.
  int last_start_y = 0, last_end_y = 0, start_y = 0, end_y = 0;
  float min_dist = 0;
  for (int x = 0; x < 256; ++x) {
    min_dist = dist.at<float>(x, 0);
    for (int y = 0; y < 256; ++y) {
      if (min_dist >= dist.at<float>(x, y)) {
        end_y = y;
        min_dist = dist.at<float>(x, y);
      }
    }
    if ((start_y != last_start_y) || (end_y != last_end_y)) {
      for (int i = start_y; i <= end_y; ++i) {
        hist_map.at<uchar>(i) = static_cast<uchar>(x);
      }
      last_start_y = start_y;
      last_end_y = end_y;
      start_y = last_end_y + 1;
    }
  }
  
  // Image pixel value mapping.
  tone_mapping->create(rows_, cols_, CV_32FC1);
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      uchar gray = image_.at<uchar>(r, c);
      float new_gray = static_cast<float>(hist_map.at<uchar>(gray)) / 255.0;
      tone_mapping->at<float>(r, c) = new_gray;
    }
  }
  return true;
}

// [Functions] Decorate the generated manga with a frame template.
bool MangaEngine::AddFrameTemplate(const string& frame_path) {
  cv::Mat frame = cv::imread(frame_path, 0);
  if (frame.empty() || manga_.empty())
    return false;
  cv::resize(frame, frame, cv::Size(cols_, rows_));
  cv::Mat float_frame;
  frame.convertTo(float_frame, CV_32FC1);
  manga_ = manga_.mul(float_frame / 255.0);
  return true;
}

// [Functions] Add text description for the current image.
// Currently, it doesn't support multi-line text input.
bool MangaEngine::AddText(const string& text, const string& dialog_path, const cv::Point2f left_top) {
  
  cv::Mat dialog_img = cv::imread(dialog_path, -1);
  if ((manga_.empty()) || (dialog_img.channels() != 4) || (left_top.x < 0) || (left_top.y < 0))
    return false;
  
  cv::Mat dialog_gray = cv::imread(dialog_path, 0);
  
  // Prepare the font.
  int fontFace = 4;
  double fontScale = 1;
  int thickness = 2;
  int baseline;
  cv::Size text_size = cv::getTextSize(text, fontFace, 1, thickness, &baseline);
  int scale = std::min(rows_ / text_size.height, cols_ / text_size.width) / 4;
  if (scale > 1) {
    fontScale = scale;
    thickness = scale;
    text_size.height *= scale;
    text_size.width *= scale;
  }
  std::vector<cv::Mat> dialog_vec;
  cv::split(dialog_img, dialog_vec);
  cv::Mat alpha = dialog_vec[3].clone();
  cv::Size dialog_size = cv::Size(text_size.width * 2, text_size.height * 4);
  cv::resize(dialog_gray, dialog_gray, dialog_size);
  cv::resize(alpha, alpha, dialog_size);
  
  // Draw the text box.
  int row_offset = left_top.y * rows_;
  int col_offset = left_top.x * cols_;
  for (int r = 0; r < dialog_size.height; ++r) {
    int r_pos = r + row_offset;
    if (r_pos >= rows_)
      break;
    for (int c = 0; c < dialog_size.width; ++c) {
      int c_pos = c + col_offset;
      if (c_pos >= cols_)
        break;
      float weight = alpha.at<uchar>(r, c) / 255.0;
      manga_.at<float>(r_pos, c_pos) = manga_.at<float>(r_pos, c_pos) * (1 - weight) +
      weight * dialog_gray.at<uchar>(r, c) / 255.0;
    }
  }
  
  // Draw the text.
  // Text drawing position (left bottom point).
  cv::Point text_pos;
  text_pos.x = col_offset + text_size.width / 2;
  text_pos.y = row_offset + text_size.height * 2;
  
  cv::putText(manga_, text, text_pos, fontFace, fontScale, cv::Scalar::all(255), thickness);
  return true;
}
