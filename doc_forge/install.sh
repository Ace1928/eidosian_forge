#!/bin/bash
# 🌀 Eidosian Installation Script for Doc Forge
# This script installs Doc Forge with full command-line access

set -e  # Exit on error

# ANSI color codes for beautiful output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Doc Forge Installation - Eidosian Documentation System    ║${NC}"
echo -e "${BLUE}╚═════════════════════════════════════════════════════════════╝${NC}"

# Make the script executable
chmod +x bin/doc-forge
echo -e "${GREEN}✓${NC} Made doc-forge script executable"

# Install the package in development mode
pip install -e .
echo -e "${GREEN}✓${NC} Installed doc-forge package"

# Create symlink in ~/.local/bin if it's in PATH and directory exists
if [[ $PATH == *"$HOME/.local/bin"* ]] && [ -d "$HOME/.local/bin" ]; then
    ln -sf "$(pwd)/bin/doc-forge" "$HOME/.local/bin/doc-forge"
    echo -e "${GREEN}✓${NC} Created symlink in ~/.local/bin"
    echo -e "${GREEN}✓${NC} You can now run 'doc-forge' directly from the command line"
else
    echo -e "${BLUE}ℹ${NC} To use doc-forge directly, do one of the following:"
    echo "  1. Add this directory to your PATH"
    echo "  2. Create a symlink to bin/doc-forge in a directory in your PATH"
    echo "  3. Continue using 'python -m doc_forge' to run"
fi

echo -e "\n${GREEN}Doc Forge installation complete!${NC}"
