//
//  MangaCollage.h
//  MangaCollage
//
//  Created by Zhipeng Wu on 2/1/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#ifndef __MangaCollage__MangaCollage__
#define __MangaCollage__MangaCollage__

#include "MangaEngine.h"
#include "CartoonEngine.h"
#include <time.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <fstream>
#include <iostream>
#include <time.h>
#include <math.h>

#define random(x) (rand() % x)

typedef cv::Rect_<float> Rect2f;

class TreeNode {
 public:
  // Default constructor:
  TreeNode(): child_type_('N'), split_type_('N'), is_leaf_(true), alpha_(0),
              position_(Rect2f()), image_index_(-1), inner_child_(-1),
              left_child_(NULL), right_child_(NULL), parent_(NULL) { }
  
  char child_type_;      // Is this node left child "l" or right child "r".
  char split_type_;      // If this node is a inner node, we set 'v' or 'h', which indicate
                         // vertical cut or horizontal cut.
  bool is_leaf_;         // Is this node a leaf node or a inner node.
  float alpha_;          // If this node is a leaf, we set actual aspect ratio of this node.
  Rect2f position_;      // The position of the node on canvas.
  int image_index_;      // If this node is a leaf, it is related with a image.
  int inner_child_;      // The number of inner nodes among its children. including itself.
                         // For leaf node, this value is 0, and for inner nodes, it is >= 1.
  TreeNode* left_child_;
  TreeNode* right_child_;
  TreeNode* parent_;
  
};

class MangaCollage {
 public:
  // Constructors/Destructors:
  
  explicit MangaCollage (const std::string& input_image_list);
  explicit MangaCollage (const std::vector<std::string> input_image_list);
  ~MangaCollage();
  
  // Manga layout generators:
  bool CreateCollage (const cv::Size2i canvas_size,
                      const int border_size,
                      const float threshold);
  // Function overloading:
  bool CreateCollage (const cv::Size2i canvas_size,
                      const int border_size) {
    float threshold = 0.1;
    return CreateCollage(canvas_size, border_size, threshold);
  }
  bool CreateCollage (const cv::Size2i canvas_size) {
    int border_size = 6;
    return CreateCollage(canvas_size, border_size);
  }
  bool CreateCollage () {
    // A4 paper (portrait) at 150dpi, width = 1240 pixel, height = 1754 pixel.
    cv::Size2i canvas_size(1240, 1754);
    return CreateCollage(canvas_size);
  }
  
  // Collage output:
  // 'type' refers to the kind of non-photorealistic features we provide.
  // 'type = 'p': Output as a photo collage.
  // 'type = 'm': Output as a manga collage.
  // 'type = 'c': Output as a cartoon collage.
  // 'type = 's': Output as a sketch collage.
  cv::Mat OutputCollage(char type, bool accurate);
  cv::Mat OutputCollage(char type) {
    return OutputCollage(type, true);
  }
  
  // Accessors:
  
  int image_num() const {
    return image_num_;
  }
  int canvas_height() const {
    return canvas_height_;
  }
  int canvas_width() const {
    return canvas_width_;
  }
  float canvas_alpha() const {
    return canvas_alpha_;
  }

 private:
  // Private member functions:
  
  // Read input images from image list.
  bool ReadImageList(std::string input_image_list);
  // Clean and release the binary_tree.
  void ReleaseTree(TreeNode* node);
  // Copy the layout solution to class member variable tree_root_.
  void CopyTree(TreeNode** to, TreeNode* from, TreeNode* parent);
  // Generate a layout solution and return the aspect ratio.
  float GenerateSolution(TreeNode* root);
  // Randomly generate a tree structure with image_num_ leaves.
  bool TreeStructure(TreeNode* root);
  // Associate tree leaves (left-to-right) with images (input order).
  bool AssociateLeaves(TreeNode* root);
  // Find the best-fit split types for inner nodes.
  float FindBestAlpha(TreeNode* root);
  // Recursive calculate all the possible aspect ratios with respect to
  // the current tree structure.
  bool RecursiveAlpha(TreeNode* root, std::vector<float>* alphas);
  // Decode the binary tree (setting the split types of inner nodes)
  bool RecursiveDecode(TreeNode* root, const std::vector<bool>& bit_index);
  // After find the best solution, calculate the position for images.
  // This is Japanses reading style (top-bottom, right-left)
  bool CalculatePositionsJP(TreeNode* node);
  // After find the best solution, calculate the position for images.
  // This is US reading style (top-bottom, left-right)
  bool CalculatePositionsUS(TreeNode* node);
  // Get all the leaves of the current tree and save them into tree_leaves_
  bool GetTreeLeaves();
  // Recursively calculate aspect ratio for all the tree nodes.
  // The return value is the aspect ratio for the node.
  float CalculateAlpha(TreeNode* node);
  
  // Private data members:
  
  // Vector containing input image paths.
  std::vector<std::string> image_path_vec_;
  // Vector containing input images.
  std::vector<cv::Mat> image_vec_;
  // Vector containing input images' aspect ratios.
  std::vector<float> image_alpha_vec_;
  // Vector containing leaf nodes of the tree.
  std::vector<TreeNode*> tree_leaves_;
  // Canvas width. (given by the user)
  int canvas_width_;
  // Canvas height. (given by the user)
  int canvas_height_;
  // Expect canvas aspect ratio. (given by the user)
  float canvas_alpha_expect_;
  // Actual canvas aspect ratio.
  float canvas_alpha_;
  // Number of images in the collage. (number of leaf nodes in the tree)
  int image_num_;
  // Border size between manga frames.
  int border_size_;
  // Full balanced binary tree for collage generation.
  TreeNode* tree_root_;
  
  // Disallow copy and assign.
  
  void operator= (const MangaCollage&);
  MangaCollage(const MangaCollage&);
};
#endif /* defined(__MangaCollage__MangaCollage__) */
