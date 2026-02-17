#!/bin/bash
# Script to download and build llama.cpp

# Ensure we're in the right base directory
BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

# 1. Clone the llama.cpp repository if it doesn't exist
if [ ! -d "doc_forge/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp doc_forge/llama.cpp
fi
cd doc_forge/llama.cpp

# 2. Create a build directory
mkdir -p build
cd build

# 3. Configure the build with cmake
cmake ..

# 4. Compile the project
cmake --build . --config Release
