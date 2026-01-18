#!/usr/bin/env bash
# 🚀 Eidosian dependency installer
# Automatically installs dependencies for all supported languages

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$REPO_ROOT/setup_log.txt"

echo "🔮 Eidosian Dependency Installer"
echo "================================"
echo "Installing all required dependencies..."

# Check for Python
if command -v python3 &> /dev/null; then
    echo "✅ Python detected. Installing Python dependencies..."
    if [ -f "$REPO_ROOT/requirements.txt" ]; then
        python3 -m pip install -r "$REPO_ROOT/requirements.txt" || { 
            echo "⚠️ Python dependency installation failed with exit code $?" >> "$LOG_FILE"
            echo "⚠️ Python dependency installation failed" 
        }
    else
        echo "⚠️ No requirements.txt found. Skipping Python dependencies."
    fi
else
    echo "⚠️ Python not found. Skipping Python dependencies."
fi

# Check for Node.js
if command -v node &> /dev/null; then
    echo "✅ Node.js detected. Installing JavaScript dependencies..."
    if [ -f "$REPO_ROOT/package.json" ]; then
        (cd "$REPO_ROOT" && npm install) || echo "⚠️  Node.js dependency installation failed" >> "$LOG_FILE"
    else
        echo "⚠️  No package.json found. Skipping JavaScript dependencies."
    fi
else
    echo "⚠️  Node.js not found. Skipping JavaScript dependencies."
fi

# Check for Go
if command -v go &> /dev/null; then
    echo "✅ Go detected. Installing Go dependencies..."
    if [ -f "$REPO_ROOT/go.mod" ]; then
        (cd "$REPO_ROOT" && go mod download) || echo "⚠️  Go dependency installation failed" >> "$LOG_FILE"
    else
        echo "⚠️  No go.mod found. Skipping Go dependencies."
    fi
else
    echo "⚠️  Go not found. Skipping Go dependencies."
fi

# Check for Rust
if command -v cargo &> /dev/null; then
    echo "✅ Rust detected. Installing Rust dependencies..."
    if [ -f "$REPO_ROOT/Cargo.toml" ]; then
        (cd "$REPO_ROOT" && cargo fetch) || echo "⚠️  Rust dependency installation failed" >> "$LOG_FILE"
    else
        echo "⚠️  No Cargo.toml found. Skipping Rust dependencies."
    fi
else
    echo "⚠️  Rust not found. Skipping Rust dependencies."
fi

echo ""
echo "✨ Dependency installation completed!"
echo "Check $LOG_FILE for any errors."
