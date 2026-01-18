# **Figlet Forge**

```ascii
  _____ _       _      _     _____
 |  ___(_) __ _| | ___| |_  |  ___|__  _ __ __ _  ___
 | |_  | |/ _` | |/ _ \ __| | |_ / _ \| '__/ _` |/ _ \
 |  _| | | (_| | |  __/ |_  |  _| (_) | | | (_| |  __/
 |_|   |_|\__, |_|\___|\__| |_|  \___/|_|  \__, |\___|
            |___/                           |___/
```

> _"Form follows function; elegance emerges from precision"_

[![Python Package](https://github.com/Ace1928/figlet_forge/actions/workflows/python-package.yml/badge.svg)](https://github.com/Ace1928/figlet_forge/actions/workflows/python-package.yml)
[![Eidosian Universal CI](https://github.com/Ace1928/figlet_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/figlet_forge/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/figlet-forge.svg)](https://badge.fury.io/py/figlet-forge)
[![Documentation Status](https://readthedocs.io/en/latest/?badge=latest)](https://figlet-forge.readthedocs.io/)
[![Forge System](https://img.shields.io/badge/Forge-System-8A2BE2)](https://github.com/Ace1928)

## Synopsis

**Figlet Forge** is an Eidosian reimplementation extending the original pyfiglet package with colorized ANSI support, Unicode rendering, and intelligent fallbacks while maintaining full backward compatibility.

Building upon pyfiglet's foundation, Figlet Forge adds robust color support, comprehensive Unicode handling, and significant performance optimizations through recursive refinement and precision engineering.

## Features

- **Full colorized ANSI code support** for vibrant text art with foreground and background colors
- **Unicode character rendering** with comprehensive mapping for global language support
- **Expanded font ecosystem** with careful attention to licensing and compatibility
- **Intelligent fallbacks** for complete backward compatibility with older systems
- **Significant performance optimizations** without sacrificing quality
- **Enhanced maintainability** through modern Python practices and typing
- **Comprehensive documentation** for all use cases
- **Full backward compatibility** with pyfiglet

## Installation

```bash
pip install figlet-forge
```

## Quick Start

### Command Line

```bash
# Basic usage
figlet_forge "Hello world"

# Select a font
figlet_forge -f slant "Cool Text"

# Use colors
figlet_forge -f big -c "RED:BLUE" "Colorful"

# Add borders
figlet_forge --border=single "Boxed Text"

# Rainbow effect
figlet_forge --color=rainbow "Rainbow"
```

### Python API

```python
from figlet_forge import Figlet
from figlet_forge import print_figlet

# Simple rendering
fig = Figlet(font="slant")
result = fig.renderText("Hello, world!")
print(result)

# Colorized output
print_figlet("Colorful Text", font="big", colors="RED:BLUE")

# Advanced features
fig = Figlet(font="standard", width=80, justify="center")
text = fig.renderText("Centered Text")
bordered = text.border(style="double")
print(bordered)
```

## Compatibility with pyfiglet

Figlet Forge is a drop-in replacement for pyfiglet:

```python
# Original pyfiglet code
from pyfiglet import Figlet
f = Figlet(font='slant')
print(f.renderText('Hello World'))

# Works exactly the same with Figlet Forge
from figlet_forge.compat import Figlet
f = Figlet(font='slant')
print(f.renderText('Hello World'))
```

## Documentation

For comprehensive documentation, visit [figlet-forge.readthedocs.io](https://figlet-forge.readthedocs.io/).

- [User Guide](https://figlet-forge.readthedocs.io/en/latest/user_guide.html)
- [API Reference](https://figlet-forge.readthedocs.io/en/latest/api.html)
- [Font Specification](https://figlet-forge.readthedocs.io/en/latest/figfont.html)

## Examples

Check the `examples/` directory for practical use cases:

- `advanced_api_usage.py`: Demonstrates sophisticated usage patterns
- `compatibility_example.py`: Shows perfect compatibility with pyfiglet
- `ascii_to_svg_example.py`: Converts ASCII art to SVG
- `eidos_figlet_forge_manifesto.py`: Showcases the Eidosian principles

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Figlet Forge is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Original pyfiglet copyright © 2007-2023 belongs to the original pyfiglet authors.
Figlet Forge extensions copyright © 2023-2025 Lloyd Handyside and Eidosian Forge contributors.

## Related Projects

- [FIGlet](http://www.figlet.org/) - The original FIGlet program
- [pyfiglet](https://github.com/pwaller/pyfiglet) - Python implementation of FIGlet

## Eidosian Integration

Figlet Forge is a component of the Eidosian Forge ecosystem, providing typography services and text crystallization tools following Eidosian principles of elegance, precision, and recursive optimization.

> _"Elegance in recursion, precision in expression."_ - Lloyd Handyside
