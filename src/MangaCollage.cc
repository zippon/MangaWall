//
//  MangaCollage.cpp
//  MangaCollage
//
//  Created by Zhipeng Wu on 2/1/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#include <memory>
#include "MangaCollage.h"

const int kMaxTreeGeneration = 100;

// Constructors:

MangaCollage::MangaCollage (const std::string& input_image_list):
    canvas_width_(-1), canvas_height_(-1), canvas_alpha_expect_(-1),
    canvas_alpha_(-1), image_num_(-1), border_size_(-1),
    tree_root_(NULL) {
  ReadImageList(input_image_list);
  image_num_ = static_cast<int>(image_vec_.size());
}

MangaCollage::MangaCollage (const std::vector<std::string> input_image_list):
    canvas_width_(-1), canvas_height_(-1), canvas_alpha_expect_(-1),
    canvas_alpha_(-1), image_num_(static_cast<int>(input_image_list.size())),
    border_size_(-1), tree_root_(NULL) {
  for (int i = 0; i < input_image_list.size(); ++i) {
    std::string img_path = input_image_list[i];
    cv::Mat img = cv::imread(img_path.c_str(), 1);
    image_vec_.push_back(img);
    float img_alpha = static_cast<float>(img.cols) / img.rows;
    image_alpha_vec_.push_back(img_alpha);
    image_path_vec_.push_back(img_path);
  } 
}

// Destructors:

MangaCollage::~MangaCollage() {
  ReleaseTree(tree_root_);
  image_vec_.clear();
  image_alpha_vec_.clear();
  image_path_vec_.clear();
}

// Manga layout generator:

bool MangaCollage::CreateCollage (const cv::Size2i canvas_size,
                                  const int border_size,
                                  const float threshold) {
  if (image_path_vec_.empty() || (image_num_ <= 0) ||
      canvas_size.width <= 0 || canvas_size.height <= 0) {
    std::cout << "error in CreateCollage..." << std::endl;
    return false;
  }
  srand(static_cast<unsigned>(time(0)));
  // Define the manga content area.
  border_size_ = border_size;
  cv::Size2i content_size(canvas_size.width - border_size_ * 2,
                          canvas_size.height - border_size_ * 2);
  canvas_alpha_expect_ = static_cast<float>(content_size.width) /
      content_size.height;
  
  // Find the optimum layout solution.
  float min_error = 100;
  int generation_counter = 0;
  do {
    TreeNode* root = new TreeNode;
    // Generate a layout solution.
    canvas_alpha_ = GenerateSolution(root);
    if (-1 == canvas_alpha_)
      return false;
    float error = fabsf(1 - canvas_alpha_ / canvas_alpha_expect_);
    if (error < min_error) {
      min_error = error;
      // Set the current best result to tree_root_;
      CopyTree(&tree_root_, root, NULL);
    }
    ReleaseTree(root);
    ++generation_counter;
  } while ((min_error > threshold) && (generation_counter <= kMaxTreeGeneration));
  
  // Calculate the image locations.
  int content_width = content_size.width;
  int content_height = static_cast<int>(content_width / canvas_alpha_);
  canvas_width_ = content_width + border_size_ * 2;
  canvas_height_ = content_height + border_size_ * 2;
  tree_root_->position_.x = border_size_;
  tree_root_->position_.y = border_size_;
  tree_root_->position_.height = content_height;
  tree_root_->position_.width = content_width;
  if (tree_root_->left_child_)
    CalculatePositionsUS(tree_root_->left_child_);
  if (tree_root_->right_child_)
    CalculatePositionsUS(tree_root_->right_child_);
  return true;
}

// Generate a layout solution and return the aspect ratio.
float MangaCollage::GenerateSolution(TreeNode* root) {
  if ((NULL == root) || (image_num_ <= 0) || (canvas_alpha_expect_ < 0)) {
    std::cout << "error in GenerateSolution..." << std::endl;
    return -1;
  }
  
  // Step 1: Randomly generate a tree structure with image_num_ leaves.
  if (!TreeStructure(root))
    return -1;
  // Step 2: Associate tree leaves (left-to-right) with images (input order).
  if (!AssociateLeaves(root))
    return -1;
  // Step 3: Find the best-fit split types for inner nodes.
  return FindBestAlpha(root);
}

// Randomly generate a tree structure with image_num_ leaves.
bool MangaCollage::TreeStructure(TreeNode* root) {
  if ((NULL == root) || (image_num_ <= 0)) {
    std::cout << "error in TreeStructure..." << std::endl;
    return false;
  }
  // Step 1: create a perfect binary tree.
  int m = image_num_;
  int depth = 0;
  while (m) {
    ++depth;
    m >>= 1;
  }
  std::queue<TreeNode*> node_queue;
  node_queue.push(root);
  int current_layer = 1;
  while (!node_queue.empty() && (current_layer < depth)) {
    ++current_layer;
    int layer_node_num = static_cast<int>(node_queue.size());
    for (int i = 0; i < layer_node_num; ++i) {
      TreeNode* node = node_queue.front();
      node_queue.pop();
      node->is_leaf_ = false;
      node->left_child_ = new TreeNode;
      node->left_child_->child_type_ = 'l';
      node->left_child_->parent_ = node;
      node_queue.push(node->left_child_);
      node->right_child_ = new TreeNode;
      node->right_child_->child_type_ = 'r';
      node->right_child_->parent_ = node;
      node_queue.push(node->right_child_);
    }
  }
  int leaf_num = static_cast<int>(pow(2, depth -1));
  assert(node_queue.size() == leaf_num);
  std::vector<TreeNode*> leaf_nodes;
  for (int i = 0; i < leaf_num; ++i) {
    leaf_nodes.push_back(node_queue.front());
    node_queue.pop();
  }
  
  // Step 2: there are image_num_ - pow(2, depth - 1) leaves uncreated.
  int left_leaves = image_num_ - leaf_num;
  bool* leaf_visited = new bool[leaf_num];
  for (int i = 0; i < leaf_num; ++i) leaf_visited[i] = false;
  int counter = 0;
  while (counter < left_leaves) {
    int rand_ind = random(left_leaves);
    if (leaf_visited[rand_ind] == true) continue;
    leaf_visited[rand_ind] = true;
    leaf_nodes[rand_ind]->is_leaf_ = false;
    leaf_nodes[rand_ind]->left_child_ = new TreeNode();
    leaf_nodes[rand_ind]->left_child_->child_type_ = 'l';
    leaf_nodes[rand_ind]->left_child_->parent_ = leaf_nodes[rand_ind];
    leaf_nodes[rand_ind]->right_child_ = new TreeNode();
    leaf_nodes[rand_ind]->right_child_->child_type_ = 'r';
    leaf_nodes[rand_ind]->right_child_->parent_ = leaf_nodes[rand_ind];
    ++counter;
  }
  delete[] leaf_visited;
  return true;
}

bool MangaCollage::AssociateLeaves(TreeNode* root) {
  if ((NULL == root) || (image_num_ <= 0)) {
    std::cout << "error in AssociateLeaves..." << std::endl;
    return false;
  }
  // Traversing the tree to visit all the leaves from left to
  // right and dispatch the images.
  int index = 0;
  std::stack<TreeNode*> node_stack;
  node_stack.push(root);
  while (!node_stack.empty()) {
    TreeNode* node = node_stack.top();
    node_stack.pop();
    if (node->is_leaf_) {
      node->image_index_ = index;
      node->alpha_ = image_alpha_vec_[index];
      ++index;
      if (index > image_num_) {
        std::cout << "error: too many tree leaves..." << std::endl;
        return false;
      }
    } else {
      if (node->right_child_) {
        node_stack.push(node->right_child_);
      }
      if (node->left_child_) {
        node_stack.push(node->left_child_);
      }
    }
  }
  if (index < image_num_) {
    std::cout << "error: too many images..." << std::endl;
    return false;
  }
  return true;
}

// Find the besr (nearest to canvas_alpha_expect_) aspect ratio for the current
// tree structure.
float MangaCollage::FindBestAlpha(TreeNode* root) {
  if ((NULL == root) || (canvas_alpha_expect_ <= 0)) {
    std::cout << "error in FindBestAlpha..." << std::endl;
    return -1;
  }
  std::vector<float> alphas;
  if (!RecursiveAlpha(root, &alphas)) {
    std::cout << "error in RecursiveAlpha..." << std::endl;
    return -1;
  }
  assert(root->inner_child_ == image_num_ - 1);
  // Find the nearest aspect ratio.
  float best_alpha = 100;
  int best_index = 0;
  for (int i = 0; i < pow(2, root->inner_child_); ++i) {
    if (fabsf(alphas[i] - canvas_alpha_expect_) <
        fabsf(best_alpha - canvas_alpha_expect_)) {
      best_alpha = alphas[i];
      best_index = i;
    }
  }
  // Based on the index, set the split type ('v' or 'h') for all the inner nodes.
  std::vector<bool> bit_index;
  std::stack<bool> bit_stack;
  for (int i = 0; i < root->inner_child_; ++i) {
    if (best_index & 1) {
      bit_stack.push(true);
    } else {
      bit_stack.push(false);
    }
    best_index >>= 1;
  }
  while (!bit_stack.empty()) {
    bool v = bit_stack.top();
    bit_index.push_back(v);
    bit_stack.pop();
  }
  assert(0 == best_index);
  if(!RecursiveDecode(root, bit_index)) {
    std::cout << "error in RecursiveDecode...." << std::endl;
    return -1;
  }
  // Now, we have constructed a binary tree and set the inner node types,
  // we then alculate the alpha for each of the tree node.
  float alpha = CalculateAlpha(root);
  return alpha;
}

bool MangaCollage::RecursiveDecode(TreeNode* root,
                                   const std::vector<bool>& bit_index) {
  if ((NULL == root) || (root->inner_child_ != bit_index.size()))
    return false;
  if (root->is_leaf_)
    return true;
  // Decode split type
  // 0: 'v' - cut
  // 1: 'h' - cut
  if (bit_index[0])
    root->split_type_ = 'h';
  else
    root->split_type_ = 'v';
  std::vector<bool> left_bit_index, right_bit_index;
  int i = 1;
  for (; i <= root->left_child_->inner_child_; ++i) {
    left_bit_index.push_back(bit_index[i]);
  }
  for (; i < root->inner_child_; ++i) {
    right_bit_index.push_back(bit_index[i]);
  }
  if (!RecursiveDecode(root->left_child_, left_bit_index))
    return false;
  if (!RecursiveDecode(root->right_child_, right_bit_index))
    return false;
  return true;
}

bool MangaCollage::RecursiveAlpha(TreeNode* root, std::vector<float>* alphas) {
  if ((NULL == root) || (NULL == alphas))
    return false;
  if (root->is_leaf_) {
    // Current node is leaf.
    root->inner_child_ = 0;
    alphas->push_back(root->alpha_);
    if (root->alpha_ <= 0) {
      std::cout << "error, bad aspect ratio value..." << std::endl;
      return false;
    }
    return true;
  }
  // Else, current node is inner node (has two children).
  std::vector<float> left_alphas, right_alphas;
  if (!RecursiveAlpha(root->left_child_, &left_alphas)){
    return false;
  }
  if (!RecursiveAlpha(root->right_child_, &right_alphas)) {
    return false;
  }
  root->inner_child_ = root->left_child_->inner_child_ +
                       root->right_child_->inner_child_ + 1;
  for (int a = 0; a < static_cast<int>(pow(2, root->left_child_->inner_child_)); ++a) {
    for (int b = 0; b < static_cast<int>(pow(2, root->right_child_->inner_child_)); ++b) {
      // We code the split types as:
      // 0 for 'v' - cut
      // 1 for 'h' - cut
      // In 'v' - cut, the result aspect ratio is the sum.
      float v_alpha = left_alphas[a] + right_alphas[b];
      alphas->push_back(v_alpha);
    }
  }
  for (int a = 0; a < static_cast<int>(pow(2, root->left_child_->inner_child_)); ++a) {
    for (int b = 0; b < static_cast<int>(pow(2, root->right_child_->inner_child_)); ++b) {
      // In 'h' - cut, the result aspect ratio is the Reciprocal sum.
      float h_alpha = (left_alphas[a] * right_alphas[b]) /
                      (left_alphas[a] + right_alphas[b]);
      alphas->push_back(h_alpha);
    }
  }
  assert(alphas->size() == static_cast<int>(pow(2, root->inner_child_)));
  return true;
}


// The images are stored in the image list, one image path per row.
// This function reads the images into image_vec_ and their aspect
// ratios into image_alpha_vec_.
bool MangaCollage::ReadImageList(std::string input_image_list) {
  std::ifstream input_list(input_image_list.c_str());
  if (!input_list) {
    std::cout << "error in ReadImageList..." << std::endl;
    return false;
  }
  
  while (!input_list.eof()) {
    std::string img_path;
    std::getline(input_list, img_path);
    // std::cout << img_path <<std::endl;
    cv::Mat img = cv::imread(img_path.c_str(), 1);
    if (img.empty())
      break;
    image_vec_.push_back(img);
    float img_alpha = static_cast<float>(img.cols) / img.rows;
    image_alpha_vec_.push_back(img_alpha);
    image_path_vec_.push_back(img_path);
  }
  input_list.close();
  return true;
}

// Release the binary tree.
void MangaCollage::ReleaseTree(TreeNode* node) {
  if (node == NULL) return;
  if (node->left_child_) ReleaseTree(node->left_child_);
  if (node->right_child_) ReleaseTree(node->right_child_);
  delete node;
}

void MangaCollage::CopyTree(TreeNode** to, TreeNode* from, TreeNode* parent) {
  if (NULL == from)
    return;
  if (*to)
    ReleaseTree(*to);
  *to = new TreeNode;
  (*to)->alpha_ = from->alpha_;
  (*to)->split_type_ = from->split_type_;
  (*to)->child_type_ = from->child_type_;
  (*to)->image_index_ = from->image_index_;
  (*to)->is_leaf_ = from->is_leaf_;
  (*to)->parent_ = parent;
  (*to)->position_ = from->position_;
  (*to)->inner_child_ = from->inner_child_;
  if (from->left_child_) {
    CopyTree(&((*to)->left_child_), from->left_child_, (*to));
  } else {
    (*to)->left_child_ = NULL;
  }
  if (from->right_child_) {
    CopyTree(&((*to)->right_child_), from->right_child_, (*to));
  } else {
    (*to)->right_child_ = NULL;
  }
}

// Top-down Calculate the image positions in the colage.
bool MangaCollage::CalculatePositionsUS(TreeNode* node) {
  // Step 1: calculate height & width.
  if (node->parent_->split_type_ == 'v') {
    // Vertical cut, height unchanged.
    node->position_.height = node->parent_->position_.height;
    if (node->child_type_ == 'l') {
      node->position_.width = node->position_.height * node->alpha_;
      // If it is left child, use its parent's x & y.
      node->position_.x = node->parent_->position_.x;
      node->position_.y = node->parent_->position_.y;
    } else if (node->child_type_ == 'r') {
      node->position_.width = node->parent_->position_.width -
      node->parent_->left_child_->position_.width;
      // y (row) unchanged, x (colmn) changed.
      node->position_.y = node->parent_->position_.y;
      node->position_.x = node->parent_->position_.x +
      node->parent_->position_.width -
      node->position_.width;
    } else {
      std::cout << "error: CalculatePositions V" << std::endl;
      return false;
    }
  } else if (node->parent_->split_type_ == 'h') {
    // Horizontal cut, width unchanged.
    node->position_.width = node->parent_->position_.width;
    if (node->child_type_ == 'l') {
      node->position_.height = node->position_.width / node->alpha_;
      // If it is left child, use its parent's x & y.
      node->position_.x = node->parent_->position_.x;
      node->position_.y = node->parent_->position_.y;
    } else if (node->child_type_ == 'r') {
      node->position_.height = node->parent_->position_.height -
      node->parent_->left_child_->position_.height;
      // x (column) unchanged, y (row) changed.
      node->position_.x = node->parent_->position_.x;
      node->position_.y = node->parent_->position_.y +
      node->parent_->position_.height -
      node->position_.height;
    } else {
      std::cout << "error: CalculatePositions H" << std::endl;
      return false;
    }
  } else {
    std::cout << "error: CalculatePositions undefiend..." << std::endl;
  }
  
  // Calculation for children.
  if (node->left_child_) {
    bool success = CalculatePositionsUS(node->left_child_);
    if (!success) return false;
  }
  if (node->right_child_) {
    bool success = CalculatePositionsUS(node->right_child_);
    if (!success) return false;
  }
  return true;
}

// Top-down Calculate the image positions in the colage.
bool MangaCollage::CalculatePositionsJP(TreeNode* node) {
  // Step 1: calculate height & width.
  if (node->parent_->split_type_ == 'v') {
    // Vertical cut, height unchanged.
    node->position_.height = node->parent_->position_.height;
    if (node->child_type_ == 'l') {
      node->position_.width = node->position_.height * node->alpha_;
      node->position_.y = node->parent_->position_.y;
      node->position_.x = node->parent_->position_.x +
      node->parent_->position_.width -
      node->position_.width;
    } else if (node->child_type_ == 'r') {
      node->position_.width = node->parent_->position_.width -
      node->parent_->left_child_->position_.width;
      node->position_.x = node->parent_->position_.x;
      node->position_.y = node->parent_->position_.y;
    } else {
      std::cout << "error: CalculatePositions V" << std::endl;
      return false;
    } 
  } else if (node->parent_->split_type_ == 'h') {
    // Horizontal cut, width unchanged.
    node->position_.width = node->parent_->position_.width;
    if (node->child_type_ == 'l') {
      node->position_.height = node->position_.width / node->alpha_;
      node->position_.x = node->parent_->position_.x;
      node->position_.y = node->parent_->position_.y;
    } else if (node->child_type_ == 'r') {
      node->position_.height = node->parent_->position_.height -
      node->parent_->left_child_->position_.height;
      node->position_.x = node->parent_->position_.x;
      node->position_.y = node->parent_->position_.y +
      node->parent_->position_.height -
      node->position_.height;
    } else {
      std::cout << "error: CalculatePositions H" << std::endl;
    }
  } else {
    std::cout << "error: CalculatePositions undefined..." << std::endl;
    return false;
  }
  
  // Calculation for children.
  if (node->left_child_) {
    bool success = CalculatePositionsJP(node->left_child_);
    if (!success) return false;
  }
  if (node->right_child_) {
    bool success = CalculatePositionsJP(node->right_child_);
    if (!success) return false;
  }
  return true;
}

cv::Mat MangaCollage::OutputCollage(char type, bool accurate) {
  cv::Mat canvas;
  if ((-1 == canvas_alpha_) || (-1 == canvas_width_) || (-1 == canvas_height_)) {
    std::cout << "error: OutputCollage..." << std::endl;
    return canvas;
  }
  if (!GetTreeLeaves())  return canvas;
  canvas.create(canvas_height_, canvas_width_, CV_8UC3);
  canvas.setTo(cv::Scalar::all(255));
  assert(image_vec_[0].type() == CV_8UC3);
  assert(tree_leaves_.size() == image_num_);
  for (int i = 0; i < image_num_; ++i) {
    // copy the image.
    int img_ind = tree_leaves_[i]->image_index_;
    Rect2f pos = tree_leaves_[i]->position_;
    if ((pos.width - border_size_ * 2 <= 0) ||
        (pos.height - border_size_ * 2 <= 0)) {
      std::cout << "error in OutputCollage..." << std::endl;
      return canvas;
    }
    cv::Rect img_inside_border(pos.x + border_size_,
                               pos.y + border_size_ ,
                               pos.width - border_size_ * 2,
                               pos.height - border_size_ * 2);
    cv::Mat roi(canvas, img_inside_border);
    cv::Mat resized_img(img_inside_border.height,
                        img_inside_border.width,
                        CV_8UC3);
    
    switch (type) {
      case 'p': {
        // Create a photo collage.
        cv::resize(image_vec_[img_ind], resized_img, resized_img.size());
        break;
      }
      case 'm': {
        // Create a manga collage.
        std::auto_ptr<MangaEngine>
        manga_engine(new MangaEngine(image_vec_[img_ind]));
        manga_engine->Convert2Manga();
        cv::Mat manga_img = manga_engine->manga() * 255;
        // manga_img is CV_32FC1 type, we have to convert it to CV_8UC1.
        manga_img.convertTo(manga_img, CV_8UC1);
        // we then convert it to CV_8UC3 (gray -> color).
        cv::cvtColor(manga_img, manga_img, CV_GRAY2BGR);
        cv::resize(manga_img, resized_img, resized_img.size());
        break;
      }
      case 'c': {
        // Create a cartoon manga.
        std::auto_ptr<CartoonEngine>
        cartoon_engine(new CartoonEngine(image_vec_[img_ind]));
        cartoon_engine->Convert2Cartoon();
        cv::Mat cartoon = cartoon_engine->cartoon();
        cv::resize(cartoon, resized_img, resized_img.size());
        break;
      }
      case 's': {
        // Create a sketch manga.
        break;
      }
      default: {
        std::cout << "error in OutputCollage... type not supported..." << std::endl;
        return canvas;
      }
    }
    resized_img.copyTo(roi);
    // draw a black border line.
    cv::rectangle(canvas, img_inside_border, cv::Scalar(0, 0, 0));
  }
  if (!accurate) {
    return canvas;
  } else {
    int accurate_width = canvas_width_;
    int accurate_height = (canvas_width_ - border_size_ * 2) /
    canvas_alpha_expect_ +
    border_size_ * 2;
    cv::Mat accurate_canvas(accurate_height, accurate_width, CV_8UC3);
    accurate_canvas.setTo(cv::Scalar::all(255));
    int w, h;
    cv::Rect rect;
    if (accurate_height >= canvas_height_) {
      w = canvas_width_;
      h = canvas_height_;
      rect = cv::Rect(0, static_cast<int>((accurate_height - h) / 2), w, h);
    } else {
      h = accurate_height;
      w = static_cast<int>(h * canvas_width_ / canvas_height_);
      cv::resize(canvas, canvas, cv::Size2i(w, h));
      rect = cv::Rect(static_cast<int>((accurate_width - w) / 2), 0, w, h);
    }
    cv::Mat m(accurate_canvas, rect);
    canvas.copyTo(m);
    return accurate_canvas;
  }
}

bool MangaCollage::GetTreeLeaves() {
  if (!tree_leaves_.empty()) {
    tree_leaves_.clear();
  }
  if (NULL == tree_root_) {
    std::cout << "error in GetTreeLeaves..." << std::endl;
    return false;
  }
  std::stack<TreeNode*> node_stack;
  node_stack.push(tree_root_);
  while (!node_stack.empty()) {
    TreeNode* node = node_stack.top();
    node_stack.pop();
    if (node->is_leaf_) {
      tree_leaves_.push_back(node);
    } else {
      node_stack.push(node->right_child_);
      node_stack.push(node->left_child_);
    }
  }
  return true;
}

float MangaCollage::CalculateAlpha(TreeNode* node) {
  if (!node->is_leaf_) {
    float left_alpha = CalculateAlpha(node->left_child_);
    float right_alpha = CalculateAlpha(node->right_child_);
    if (node->split_type_ == 'v') {
      node->alpha_ = left_alpha + right_alpha;
      return node->alpha_;
    } else if (node->split_type_ == 'h') {
      node->alpha_ = (left_alpha * right_alpha) / (left_alpha + right_alpha);
      return node->alpha_;
    } else {
      std::cout << "error in CalculateAlpha..." << std::endl;
      return -1;
    }
  } else {
    // This is a leaf node, just return the image's aspect ratio.
    return node->alpha_;
  }
}
