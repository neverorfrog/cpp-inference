#!/bin/zsh

pip install -r requirements.txt

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
external="${PWD}/external"
mkdir -p ${external}

cd ${external}
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest -d .
rm -rf libtorch-shared-with-deps-latest.zip

cd executorch
git checkout v0.2.0
git submodule sync
git submodule update --init
./install_requirements.sh

# Remove existing cmake-out directory
sudo rm -rf cmake-out

# Create cmake-out directory
mkdir cmake-out
cd cmake-out

# Run CMake configuration
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/home/neverorfrog/.miniconda3/envs/executorch/bin/python3.11 \
    -DBUCK2=${SCRIPT_DIR}/external/buck2 \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON 

# Check if CMake configuration was successful
if [ $? -eq 0 ]; then
    # Build the project
    sudo cmake --build . -j13 --target install --config Release
else
    echo "CMake configuration failed. Exiting..."
fi
