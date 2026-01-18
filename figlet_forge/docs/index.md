# Figlet Forge

**Figlet Forge** is an Eidosian reimplementation extending the original pyfiglet
package with colorized ANSI support, Unicode rendering, and intelligent fallbacks.

![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)

## Features

- **Full colorized ANSI code support** for vibrant text art with foreground and background colors
- **Unicode character rendering** with comprehensive mapping for global language support
- **Expanded font ecosystem** with careful attention to licensing and compatibility
- **Intelligent fallbacks** for complete backward compatibility with older systems
- **Significant performance optimizations** without sacrificing quality
- **Enhanced maintainability** through modern Python practices and typing
- **Comprehensive documentation** for all use cases
- **Full backward compatibility** with pyfiglet

## Getting Started

### Installation

```bash
pip install figlet-forge
```

### Basic Usage

```python
from figlet_forge import Figlet
from figlet_forge import print_figlet

# Simple rendering
fig = Figlet(font="slant")
result = fig.renderText("Hello, world!")
print(result)

# Colorized output
print_figlet("Colorful Text", font="big", colors="RED:BLUE")
```

## Advanced Features

### Color Support

Figlet Forge supports ANSI color codes for both foreground and background:

```python
# Syntax: foreground:background
print_figlet("Warning!", font="standard", colors="RED:YELLOW")

# Foreground only
print_figlet("Success", font="small", colors="GREEN")

# RGB support (for terminals that support it)
print_figlet("Custom", font="slant", colors="255;128;0:0;0;128")
```

### Unicode Support

Render text in any language:

```python
# Japanese text
print_figlet("こんにちは世界", font="standard")

# Arabic text (right-to-left)
fig = Figlet(font="standard", direction="right-to-left")
print(fig.renderText("مرحبا بالعالم"))
```

### Layout Control

Control the spacing and arrangement of your figlet text:

```python
# Full width rendering
fig = Figlet(font="standard", width=80, justify="center")
print(fig.renderText("Centered Text"))

# Smushed rendering (compact)
fig = Figlet(font="standard", width=80)
print(fig.renderText("Compact Text"))
```

### Font Management

```python
from figlet_forge import Figlet

# List available fonts
fonts = Figlet().getFonts()
print(fonts)

# Get information about a specific font
font_info = Figlet.getFontInfo("slant")
print(font_info)

# Install a new font
Figlet.installFont("/path/to/myfont.flf")
```

## Command Line Interface

Figlet Forge provides a powerful command-line interface:

```bash
# Basic usage
figlet_forge "Hello world"

# Select a font
figlet_forge -f slant "Cool Text"

# Use colors
figlet_forge -f big -c "RED:BLUE" "Colorful"

# List available fonts
figlet_forge -l

# Get information about a font
figlet_forge -i -f standard

# Control width and justification
figlet_forge -w 60 -j center "Centered"

# Unicode support
figlet_forge -u "こんにちは世界"
```

## Differences from pyfiglet

While maintaining full backward compatibility with pyfiglet, Figlet Forge adds:

1. **Color support**: The original doesn't support ANSI color codes
2. **Improved Unicode handling**: Better support for non-Latin scripts
3. **Performance optimizations**: Faster rendering, especially for large texts
4. **Type annotations**: Full typing support for modern Python development
5. **Enhanced API**: More intuitive and flexible API design
6. **Intelligent fallbacks**: Gracefully handles edge cases

## API Reference

### Core Classes

- `Figlet` - The main class for rendering figlet text
- `FigletString` - Represents rendered figlet output with transformation methods
- `ColorMode` - Enumeration for color handling options
- `ColorScheme` - Class for defining custom color schemes

### Key Methods

- `Figlet.renderText(text)` - Renders text in the selected font
- `Figlet.getFonts()` - Returns list of available fonts
- `Figlet.setFont(font)` - Changes the current font
- `print_figlet(text, **options)` - Convenience function for quick colored output
- `FigletString.reverse()` - Reverses rendered text horizontally
- `FigletString.flip()` - Flips rendered text vertically

## Contributing

Contributions are welcome! Please check our [GitHub repository](https://github.com/Ace1928/figlet_forge) for:

1. Issue reporting
2. Feature requests
3. Pull requests
4. Documentation improvements

## License

Figlet Forge is licensed under the MIT License - see the [LICENSE](https://github.com/Ace1928/figlet_forge/blob/main/LICENSE) file for details.

Original pyfiglet copyright © 2007-2023 belongs to the original pyfiglet authors.
Figlet Forge extensions copyright © 2023-2025 Lloyd Handyside and Eidosian Forge contributors.
