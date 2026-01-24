#!/bin/bash
# Install dependencies for computer_control_forge
# Run with: bash install_deps.sh

set -e

echo "🎮 Installing Computer Control Forge dependencies..."

# System packages
echo "📦 Installing system packages (requires sudo)..."
sudo apt update
sudo apt install -y xdotool scrot

# Python packages in eidosian_venv
echo "🐍 Installing Python packages..."
cd /home/lloyd/eidosian_forge
source eidosian_venv/bin/activate

pip install pynput pillow mss

echo "✅ Dependencies installed!"
echo ""
echo "To test, run:"
echo "  cd /home/lloyd/eidosian_forge"
echo "  source eidosian_venv/bin/activate"
echo "  python -c 'from computer_control_forge import is_kill_switch_active; print(\"OK\")'"
