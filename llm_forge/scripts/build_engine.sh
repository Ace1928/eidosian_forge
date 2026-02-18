#!/bin/bash
# Eidosian build script for LLM Forge Engine (llama.cpp)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENDOR_DIR="${FORGE_ROOT}/llm_forge/vendor/llama.cpp"

cd "$VENDOR_DIR" || exit 1

echo "--- 🛠️ Building LLM Forge Engine (llama.cpp) ---"

# 1. Prepare build directory
mkdir -p build
cd build

# 2. Configure with CMake
# Using CPU-only optimizations for standard Termux environment
cmake .. -DCMAKE_BUILD_TYPE=Release 
         -DLLAMA_NATIVE=ON 
         -DLLAMA_LTO=ON

# 3. Compile with multiple cores
CORES=$(nproc)
echo "Compiling with $CORES cores..."
cmake --build . --config Release -j "$CORES"

if [ $? -eq 0 ]; then
    echo "--- ✅ Build Successful ---"
    # Create symlink to bin for easier access
    cd "${FORGE_ROOT}/llm_forge"
    ln -sfn vendor/llama.cpp/build/bin bin
else
    echo "--- ❌ Build Failed ---"
    exit 1
fi
