# Glyph Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Pixels to Symbols.**

## 🖼️ Overview

`glyph_forge` converts images and videos into high-fidelity ASCII/ANSI art.
It uses `pillow` for image processing and `numpy` for efficient pixel mapping.

## 🏗️ Architecture
- `src/glyph_forge/`: Core logic.
    - `converter.py`: Image to text transformation.
    - `cli.py`: Command line interface.

## 🚀 Usage

```bash
glyph-forge convert my_image.png --width 80 --color
```