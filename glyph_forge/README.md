# Glyph Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: 325 Passing](https://img.shields.io/badge/tests-325%20passing-brightgreen.svg)](tests/)

**Pixels to Symbols. Where images transcend their digital boundaries.**

## 🖼️ Overview

`glyph_forge` converts images and videos into high-fidelity ASCII/ANSI art with unprecedented control and quality.
It features configurable color modes (ANSI16/256/Truecolor/HTML), dithering algorithms (Floyd-Steinberg/Atkinson),
image enhancement filters, real-time streaming, and a comprehensive TUI interface.

### 🔥 Performance (January 2026)

| Mode | 480p | 720p | Notes |
|------|------|------|-------|
| Gradient | **74 fps** | **34 fps** | Exceeds 60fps target |
| Braille (plain) | **144 fps** | **64 fps** | Ultra-fast monochrome |
| Braille (ANSI256) | **33 fps** | 15 fps | Full color at speed |

## 🏗️ Architecture

- `src/glyph_forge/`: Core logic
  - `api/`: Public API (glyph_api.py)
  - `cli/`: Command-line interface (bannerize, imagize, stream)
  - `core/`: Banner generation, style management
  - `services/`: Image conversion, streaming, video processing
  - `transformers/`: Image analysis (color, depth, edges)
  - `renderers/`: Output formatters (Text, ANSI, HTML, SVG)
  - `ui/`: Textual TUI (tui.py, glyph_forge.css)
  - `config/`: Settings management

## 🚀 Quick Start

### Installation

```bash
cd eidosian_forge/glyph_forge
pip install -e .
```

### Basic Commands

```bash
# Display banner
glyph-forge

# Convert an image
glyph-forge imagize convert my_image.png --width 80

# Generate a text banner
glyph-forge bannerize "Hello World" --font slant

# Launch interactive TUI
glyph-forge interactive

# Stream video/webcam
glyph-forge stream --webcam
glyph-forge stream https://youtube.com/watch?v=VIDEO_ID
```

## 📸 Image Conversion

### Basic Usage

```bash
# Grayscale conversion
glyph-forge imagize convert image.png --width 100

# With color output
glyph-forge imagize convert image.png --color truecolor

# ANSI 256-color mode
glyph-forge imagize convert image.png --color ansi256
```

### Quality Controls

```bash
# Full quality pipeline
glyph-forge imagize convert image.png \
  --color truecolor \
  --dither --dither-algorithm atkinson \
  --gamma 1.1 \
  --autocontrast \
  --equalize \
  --edge-enhance \
  --sharpen \
  --resample lanczos
```

### All Image Options

| Option | Description | Default |
|--------|-------------|---------|
| `--width` | Output width in characters | auto |
| `--color` | Color mode (none/ansi16/ansi256/truecolor/html) | none |
| `--charset` | Character set (standard/blocks/detailed/etc) | standard |
| `--dither` | Enable dithering | false |
| `--dither-algorithm` | Dither algorithm (floyd-steinberg/atkinson) | floyd-steinberg |
| `--gamma` | Gamma correction (0.1-3.0) | 1.0 |
| `--autocontrast` | Auto-adjust contrast | false |
| `--equalize` | Histogram equalization | false |
| `--edge-enhance` | Edge enhancement | false |
| `--sharpen` | Sharpening | false |
| `--blur-radius` | Blur radius | 0.0 |
| `--posterize-bits` | Posterize bits (1-8) | 8 |
| `--invert` | Invert image | false |
| `--resample` | Resample filter (nearest/bilinear/bicubic/lanczos) | bilinear |

## 🎬 High-Performance Streaming

### New Modular Streaming Engine

Glyph Forge includes a high-performance modular streaming engine with:
- **Ultra Performance**: 74 fps @ 480p, 34 fps @ 720p (gradient mode)
- **Braille Sub-Pixel**: 2x4 dots per character = 8x effective resolution
- **ANSI256 Optimization**: 1.6x faster than TrueColor with RLE compression
- **Pre-buffering**: Smooth playback by buffering frames before display
- **Source FPS Matching**: Automatically matches source video frame rate
- **Audio Synchronization**: Optional audio playback (requires pygame/simpleaudio)
- **Quality Presets**: MINIMAL/LOW/STANDARD/HIGH/MAXIMUM

### CLI Commands

```bash
# Stream video file
glyph-forge stream video.mp4

# Stream YouTube video
glyph-forge stream https://youtube.com/watch?v=VIDEO_ID

# Stream webcam
glyph-forge stream --webcam 0

# High quality with audio
glyph-forge stream video.mp4 --quality high --audio --prebuffer 60

# Low latency streaming (minimal buffering)
glyph-forge stream --webcam 0 --prebuffer 5 --quality low

# Use legacy glyph_stream.py
glyph-forge stream video.mp4 --legacy
```

### Stream Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fps` | Target FPS (None = match source) | auto |
| `--scale` | Scale factor (1-4) | 1 |
| `--quality` | Quality preset (minimal/low/standard/high/maximum) | standard |
| `--gradient` | Character gradient (standard/enhanced/braille/ascii) | standard |
| `--algorithm` | Edge detection (sobel/prewitt/scharr/laplacian/canny) | sobel |
| `--color/--no-color` | Enable ANSI color | true |
| `--audio/--no-audio` | Enable audio playback | true |
| `--stats/--no-stats` | Show performance statistics | true |
| `--border/--no-border` | Show decorative border | true |
| `--adaptive` | Enable adaptive quality adjustment | false |
| `--prebuffer` | Prebuffer seconds before playback | 3.0 |
| `--buffer` | Total buffer size in seconds | 30.0 |
| `--record` | Record stream to video file | None |
| `--legacy` | Use legacy glyph_stream.py | false |

### 🚀 Premium Stream Mode (Default)

The default `glyph-forge stream` command now uses the premium engine with:

- **1080p Resolution** (auto-downscales if source is lower)
- **60 FPS Target** (matches source FPS if lower)
- **Braille Sub-Pixel Rendering** (2x4 dots = 8 pixels per character)
- **ANSI256 Color** (fast color with 216 colors + 24 grayscale)
- **30 Second Buffer** (~1800 frames at 60fps)
- **Audio Enabled** (requires pygame/simpleaudio)
- **Smart Buffering** (caps at stream length for short videos)
- **Optional Recording** (save stream to video file)

```bash
# Premium defaults (1080p, 60fps, braille, color, buffered, audio)
glyph-forge stream video.mp4

# With recording
glyph-forge stream video.mp4 --record output.mp4

# YouTube with recording
glyph-forge stream https://youtube.com/watch?v=VIDEO_ID --record youtube.mp4

# Maximum quality (TrueColor - slower but 16M colors)
glyph-forge stream video.mp4 --color truecolor --mode hybrid

# Lower resolution for lower-powered machines
glyph-forge stream video.mp4 --resolution 480p --fps 30

# Webcam streaming
glyph-forge stream --webcam 0 --buffer 5.0 --prebuffer 0.5
```

### Python API

```python
from glyph_forge.streaming import stream, stream_youtube, stream_webcam

# Simple streaming
stream("video.mp4")
stream("https://www.youtube.com/watch?v=...")
stream_webcam(0)

# With options
stream("video.mp4", quality="high", audio=True, prebuffer=60)

# Full control with StreamEngine
from glyph_forge.streaming import StreamEngine, StreamConfig, QualityLevel

config = StreamConfig(
    quality_level=QualityLevel.HIGH,
    color_enabled=True,
    audio_enabled=True,
    prebuffer_frames=60,
    adaptive_quality=False,  # Disabled by default
)

engine = StreamEngine(config)
engine.run("video.mp4")

# Premium streaming with recording
from glyph_forge.streaming.premium import PremiumConfig, PremiumStreamEngine

config = PremiumConfig(
    resolution="1080p",
    target_fps=60,
    render_mode="braille",
    color_mode="ansi256",
    buffer_seconds=30.0,
    record_enabled=True,
    record_path="output.mp4",
)
engine = PremiumStreamEngine(config)
engine.run("video.mp4")
```

### Streaming Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `opencv-python` | Video capture | Yes |
| `yt-dlp` | YouTube extraction | For YouTube |
| `pygame` or `simpleaudio` | Audio playback | Optional |
| `ffmpeg` | Audio extraction | Optional |
| `mss` | Screen capture | For browser |
| `Xvfb` | Virtual display | For browser (Linux) |

## 🎨 Text Banners

```bash
# Basic banner
glyph-forge bannerize "HELLO" --font slant

# With style
glyph-forge bannerize "GLYPH" --font banner3 --style boxed

# With effects
glyph-forge bannerize "FORGE" --font cyberlarge --effect glow --color

# List available fonts
glyph-forge bannerize list-fonts

# Preview a font
glyph-forge bannerize preview --font slant
```

### Banner Fonts (50+)

slant, banner, banner3, big, block, bubble, digital, doom, epic, isometric1,
larry3d, letters, ogre, roman, script, shadow, small, smslant, standard, starwars...

### Banner Styles

- `minimal`: Clean, no borders
- `boxed`: Single-line box
- `double`: Double-line box
- `shadowed`: With shadow effect
- `metallic`: Metallic appearance
- `circuit`: Digital circuit style
- `eidosian`: Full Eidosian treatment

### Banner Effects

- `shadow`: Drop shadow
- `glow`: Glow effect
- `emboss`: Embossed 3D
- `digital`: Digital/scanline
- `fade`: Gradient fade

## 🖥️ Interactive TUI

Launch the full-featured terminal UI:

```bash
glyph-forge interactive
```

### TUI Features

- **Banner Tab**: Generate banners with all fonts/styles/effects
- **Image Tab**: Convert images with full parameter control
- **Streaming Tab**: Launch YouTube/webcam/browser streams
- **Settings Tab**: Configure defaults, save/reset preferences
- **About Tab**: View banner, shortcuts, dependency status

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Navigate between tabs |
| `Enter` | Execute/select |
| `Escape` | Cancel/close |
| `Ctrl+C` | Quit |

## 🐍 Python API

```python
from glyph_forge import get_api

api = get_api()

# Generate a banner
banner = api.generate_banner("Hello", font="slant", style="boxed")
print(banner)

# Convert an image
from PIL import Image
img = Image.open("photo.jpg")
ascii_art = api.image_to_glyph(img, width=80, color_mode="truecolor")
print(ascii_art)

# Preview fonts
fonts = api.preview_font("Test", font_names=["slant", "banner", "doom"])
for font_name, preview in fonts.items():
    print(f"--- {font_name} ---")
    print(preview)
```

## ⚙️ Configuration

### Config Location

- User config: `~/.config/glyph_forge/config.json`
- System config: `/etc/glyph_forge/config.json`

### Example Config

```json
{
  "banner": {
    "default_font": "slant",
    "default_style": "minimal",
    "cache_enabled": true
  },
  "image": {
    "default_width": 80,
    "default_charset": "standard",
    "default_color_mode": "truecolor",
    "gamma": 1.0,
    "dithering": false
  },
  "performance": {
    "use_numpy_optimization": true,
    "enable_caching": true
  }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=glyph_forge

# Run specific test file
pytest tests/test_tui.py -v
```

**Current Status**: 197 tests passing

## 📦 Dependencies

### Required

- Python ≥3.10
- Pillow ≥9.0.0
- NumPy ≥1.26.0
- PyFiglet ≥0.8.0
- Rich ≥13.7.0
- Typer ≥0.9.0
- Textual ≥0.75.0

### Optional

- opencv-python (video/webcam)
- yt-dlp (YouTube)
- mss (screen capture)

## 📚 Documentation

- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Detailed project status
- [INSTALL.md](INSTALL.md) - Installation guide
- [TODO.md](TODO.md) - Development roadmap
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

---

**Glyph Forge** - *Fully Eidosian. Zero compromise.*
