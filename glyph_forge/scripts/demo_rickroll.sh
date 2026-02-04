#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${ROOT_DIR}/assets"
VIDEO_URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
OUTPUT_PATH="${ASSETS_DIR}/rickroll.mp4"

mkdir -p "${ASSETS_DIR}"

if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "yt-dlp is required. Install with: pip install yt-dlp"
  exit 1
fi

if [ ! -f "${OUTPUT_PATH}" ]; then
  echo "Downloading Rickroll video..."
  yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" -o "${ASSETS_DIR}/rickroll.%(ext)s" "${VIDEO_URL}"
fi

echo "Local demo (high quality):"
glyph-forge stream "${OUTPUT_PATH}" --resolution 720p --fps 30 --mode braille --color ansi256 --audio --stats

echo "YouTube demo (direct stream):"
glyph-forge stream "${VIDEO_URL}" --resolution 720p --fps 30 --mode braille --color ansi256 --audio --stats

echo "Render-then-play demo (muxed audio):"
glyph-forge stream "${OUTPUT_PATH}" --render-play --resolution 720p --fps 30 --mode braille --color ansi256
