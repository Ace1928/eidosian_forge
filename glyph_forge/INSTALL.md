# Glyph Forge Installation Guide

## Quick Install

```bash
# From the eidosian_forge root
pip install -e ./glyph_forge
```

## Dependencies

- Python >=3.12
- pillow >=9.0.0
- numpy >=1.26.0
- rich >=13.0.0
- typer >=0.9.0

### Optional Dependencies

For full interactive TUI experience:
```bash
pip install textual
```

For video processing:
```bash
pip install opencv-python
```

## Verify Installation

```bash
glyph-forge version
```

## CLI Usage

### Basic Commands

```bash
# Show help
glyph-forge --help

# Show version
glyph-forge version

# List all commands
glyph-forge list-commands
```

### Image to ASCII Art

```bash
# Convert image to ASCII
glyph-forge imagize convert image.png

# With custom width
glyph-forge imagize convert image.png --width 100

# With specific character set
glyph-forge imagize convert image.png --charset detailed

# Save to file
glyph-forge imagize convert image.png --output art.txt
```

### Text Banners

```bash
# Generate text banner
glyph-forge bannerize text "EIDOS"

# With custom font
glyph-forge bannerize text "EIDOS" --font slant

# With custom width
glyph-forge bannerize text "EIDOS" --width 120
```

### Interactive Mode

```bash
# Launch TUI (requires textual)
glyph-forge interactive
```

## Bash Completion

Add to your `~/.bashrc`:
```bash
source /path/to/eidosian_forge/glyph_forge/completions/glyph-forge.bash
```

## Python API

```python
from glyph_forge import image_to_glyph, text_to_banner

# Convert image
result = image_to_glyph("image.png", width=100)
print(result)

# Create banner
banner = text_to_banner("EIDOS", font="slant")
print(banner)
```

## Configuration

Settings are stored in `~/.config/glyph_forge/config.yaml`:

```yaml
banner:
  default_font: slant
  default_width: 80

image:
  default_charset: general
  default_width: 100

io:
  color_output: true
```

## Integration with Eidosian Forge

When installed as part of the Eidosian Forge ecosystem:

```bash
# Via central hub
eidosian glyph --help

# Direct CLI
glyph-forge --help
```

## Troubleshooting

### Import Error
Ensure PYTHONPATH includes the forge:
```bash
export PYTHONPATH="$PYTHONPATH:/path/to/eidosian_forge/lib:/path/to/eidosian_forge/glyph_forge/src"
```

### Missing PIL/Pillow
```bash
pip install pillow
```

### Textual not found (for interactive mode)
```bash
pip install textual
```

---

*Part of the Eidosian Forge - Where pixels become characters and images transcend their digital boundaries.*
