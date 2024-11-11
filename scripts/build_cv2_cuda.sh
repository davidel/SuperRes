#!/bin/bash

set -ex

TOPDIR=$(pwd)

mkdir -p opencv_build
pushd opencv_build

SCRDIR=$(pwd)

git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib

mkdir build
pushd build

cmake \
  -DOPENCV_EXTRA_MODULES_PATH="$SCRDIR/opencv_contrib/modules" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_CUDA=ON \
  -DWITH_CUBLAS=ON \
  -DWITH_CUDNN=ON \
  -DOPENCV_DNN_CUDA=ON \
  "$SCRDIR/opencv"

make -j8 install

cp lib/python3/cv2.*.so "$TOPDIR"

popd
popd

