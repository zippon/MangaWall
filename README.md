# Short Description

![MangaWall](https://github.com/zippon/MangaWall/wiki/images/us_manga.jpg)
##Overview
The files in this folder simply allows you to create an manga collage with a set of input images. We features in the following points:

1. Fast and better manga-like rendering performance.
2. Automatic manga conversion and layout generation.
3. User-specified canvas size without cropping or changing the aspect ratio of the original images.

##Build
To build the binary, you need to pre-install [OpenCV](http://opencv.org/) on your machine.

    gcc -o MangaWall main.cc MangaCollage.cc MangaEngine.cc CartoonEngine.cc -I path/to/your/opencv/include -L path/to/your/opencv/lib -lopencv_highgui -lopencv_core -lopencv_imgproc

##Build by [CMake](http://www.cmake.org/)
Git Clone the files on your local disk. Under folder 'MangaWall':

    mkdir build
    cd build
    cmake ..
    make
Then, the binary is built at ./build/bin/MangaWall. You can test the collage:

    cd ..
    sh run_test.sh
##Test

The binary requires a list which contains a set of images. A typical example for the input images and lists can be found in the ‘*test*’ folder. To run the binary:

`./MangaWall the/path/to/your/image/list`

Then, you are required to enter the expected **width** and **height** for the manga collage canvas.

##Contact

