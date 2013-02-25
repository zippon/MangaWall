//
//  main.cpp
//  MangaCollage
//
//  Created by Zhipeng Wu on 2/1/13.
//  Copyright (c) 2013 Zhipeng Wu. All rights reserved.
//

#include <memory>
#include "MangaCollage.h"
#include <iostream>

int main(int argc, const char * argv[])
{

  if (argc != 2) {
    std::cout << "Error number of input arguments..." << std::endl;
    return -1;
  }
  std::string img_list(argv[1]);
  int canvas_w, canvas_h;
  std::cout << "Please input the canvas width..." << std::endl;
  do {
    std::cout << "canvas width [100, 10000]: ";
    std::cin >> canvas_w;
    std::cout << std::endl;
  } while ( (canvas_w < 100) || (canvas_w > 10000) );
  std::cout << "Please input the canvas height..." << std::endl;
  do {
    std::cout << "canvas height [100, 10000]: ";
    std::cin >> canvas_h;
    std::cout << std::endl;
  } while ( (canvas_h < 100) || (canvas_h > 10000) );
  int border = std::min(canvas_w, canvas_h) / 50;
  
  std::auto_ptr<MangaCollage> my_collage(new MangaCollage(img_list));
  clock_t start, end;
  start = clock();
  my_collage->CreateCollage(cv::Size2i(canvas_w, canvas_h), border);
  cv::Mat collage = my_collage->OutputCollage('m');
  end = clock();
  cv::imshow("collage", collage);
  cv::waitKey();
  std::cout << "processing time: " << (end - start) * 1000 / CLOCKS_PER_SEC
  << " ms" << std::endl;
  return 0;
}

