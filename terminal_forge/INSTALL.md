# Terminal Forge Installation Guide

## Quick Install

```bash
# From the eidosian_forge root
pip install -e ./terminal_forge
```

## Dependencies

- Python >=3.12

## Verify Installation

```bash
terminal-forge status
```

## CLI Usage

### Show Status

```bash
terminal-forge status
terminal-forge status --json
```

### Colors

```bash
# List available colors
terminal-forge colors

# Show color samples
terminal-forge colors --show
```

### Themes

```bash
# List available themes
terminal-forge themes

# Apply a theme
terminal-forge themes --apply dark
```

### Banners

```bash
# Create a simple banner
terminal-forge banner "Hello World"

# With title
terminal-forge banner "Content" --title "My Banner"

# With border style
terminal-forge banner "Text" --border double
```

### ASCII Art

```bash
# Convert image to ASCII
terminal-forge ascii image.png

# With custom width
terminal-forge ascii image.png --width 100

# With specific charset
terminal-forge ascii image.png --charset blocks
```

### Layout

```bash
# Show layout capabilities
terminal-forge layout
```

## Bash Completion

Add to your `~/.bashrc`:
```bash
source /path/to/eidosian_forge/terminal_forge/completions/terminal-forge.bash
```

## Python API

```python
from terminal_forge.colors import Color
from terminal_forge.themes import Theme
from terminal_forge.banner import Banner

# Use colors
print(f"{Color.GREEN}Success!{Color.RESET}")

# Apply theme
Theme.apply("dark")

# Create banner
b = Banner()
b.add_line("EIDOS")
b.set_border("double")
b.display()
```

## Integration with Eidosian Forge

When installed as part of the Eidosian Forge ecosystem:

```bash
# Via central hub
eidosian terminal --help

# Direct CLI
terminal-forge --help
```

---

*Part of the Eidosian Forge - Terminal styling, colors, themes, and ASCII art.*
