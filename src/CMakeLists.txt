ADD_EXECUTABLE(MangaWall main.cc MangaCollage.cc MangaEngine.cc CartoonEngine.cc)

#### Required
#    FIND_PACKAGE(OpenCV REQUIRED core highgui)
FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
   INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
   LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
   TARGET_LINK_LIBRARIES(MangaWall opencv_core opencv_highgui opencv_imgproc)
ELSE(OpenCV_FOUND)
   MESSAGE(FATAL_ERROR ”OpenCV library not found”)
ENDIF(OpenCV_FOUND)

