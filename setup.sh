sudo apt install \
  git \
  build-essential \
  cmake \
  libx11-dev \
  mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev \
  libxrandr-dev \
  libxi-dev \
  libxmu-dev \
  libblas-dev \
  libxinerama-dev \
  libxcursor-dev

cd ./libigl_manage/Lib/libigl

# rm -rf build
mkdir build
cd build

cmake ..
make -j
