# Figlet Forge User Guide

> _"Form follows function; elegance emerges from precision"_

## Introduction

Figlet Forge is an advanced FIGlet text rendering engine that transforms ordinary text into ASCII art typography. It follows the Eidosian principles of elegance, precision, and recursive optimization to provide a powerful yet intuitive interface for creating stunning textual displays.

This guide will walk you through all aspects of using Figlet Forge, from basic usage to advanced techniques and customization.

## Installation

```bash
pip install figlet-forge
```

## Basic Usage

### Command Line

The simplest way to use Figlet Forge is through the command line:

```bash
figlet_forge "Hello World"
```

This will render "Hello World" using the default font:

```
 _   _      _ _        __        __         _     _
| | | | ___| | | ___   \ \      / /__  _ __| | __| |
| |_| |/ _ \ | |/ _ \   \ \ /\ / / _ \| '__| |/ _` |
|  _  |  __/ | | (_) |   \ V  V / (_) | |  | | (_| |
|_| |_|\___|_|_|\___/     \_/\_/ \___/|_|  |_|\__,_|
```

### Python API

You can also use Figlet Forge as a Python library:

```python
from figlet_forge import Figlet

# Create a Figlet instance with your preferred settings
fig = Figlet(font="slant")

# Render text to ASCII art
result = fig.renderText("Hello World")

# Print the result
print(result)
```

This produces:

```
    __  __     ____                       __     __
   / / / /__  / / /___     _      ______  / /____/ /
  / /_/ / _ \/ / / __ \   | | /| / / __ \/ / ___/ /
 / __  /  __/ / / /_/ /   | |/ |/ / /_/ / / /  /_/
/_/ /_/\___/_/_/\____/    |__/|__/\____/_/_/  (_)
```

## Fonts

Figlet Forge comes with a variety of fonts. You can list all available fonts using:

```bash
figlet_forge --list-fonts
```

To use a specific font:

```bash
figlet_forge --font=slant "Hello"
```

```
    __  __     ____
   / / / /__  / / /___
  / /_/ / _ \/ / / __ \
 / __  /  __/ / / /_/ /
/_/ /_/\___/_/_/\____/
```

## Color Support

Figlet Forge extends the original FIGlet with rich color capabilities:

```bash
figlet_forge --color=RED "Colorful Text"
```

You can specify foreground and background colors:

```bash
figlet_forge --color=WHITE:BLUE "White on Blue"
```

### Color Effects

Several special color effects are available:

#### Rainbow

```bash
figlet_forge --color=rainbow "Rainbow"
```

The text will display with colors flowing through the rainbow spectrum.

#### Gradient

```bash
figlet_forge --color=red_to_blue "Gradient"
```

This creates a smooth color transition from one color to another.

## Text Transformation

### Justification

Control text alignment:

```bash
figlet_forge --justify=center "Centered Text" --width=80
```

```bash
figlet_forge --justify=right "Right Aligned" --width=80
```

### Flip and Reverse

```bash
figlet_forge --flip "Upside Down"
```

```bash
figlet_forge --reverse "Mirrored"
```

### Borders and Effects

Add borders around your text:

```bash
figlet_forge --border=single "Bordered Text"
```

```
┌────────────────────────────────────────────────────────┐
│  ____              _               _   _____         _  │
│ | __ )  ___  _ __ __| | ___ _ __ ___  __| |_   _____  _| |_  │
│ |  _ \ / _ \| '__/ _` |/ _ \ '__/ _ \/ _` \ \ / / _ \| | __| │
│ | |_) | (_) | | | (_| |  __/ | |  __/ (_| |\ V / (_) | | |_  │
│ |____/ \___/|_|  \__,_|\___|_|  \___|\__,_| \_/ \___/|_|\__| │
│                                                      │
└────────────────────────────────────────────────────────┘
```

Add a shadow effect:

```bash
figlet_forge --shade "Shadow"
```

## Advanced Features

### Unicode Support

Figlet Forge provides enhanced Unicode support:

```bash
figlet_forge --unicode "こんにちは世界"
```

### Custom Font Installation

Install new fonts easily:

```bash
figlet_forge --install-font my_custom_font.flf
```

### Combining Features

Features can be combined for powerful effects:

```bash
figlet_forge --font=big --color=rainbow --border=double --justify=center "Awesome!"
```

## Programmatic Usage

### Basic Rendering

```python
from figlet_forge import Figlet

# Create a Figlet instance
fig = Figlet(font="standard", width=80, justify="center")

# Render text
text_art = fig.renderText("Python API")

# Print result
print(text_art)
```

### With Color

```python
from figlet_forge import Figlet
from figlet_forge.color.effects import rainbow_colorize

# Render text
fig = Figlet(font="slant")
text_art = fig.renderText("Colorful")

# Apply rainbow effect
colored_art = rainbow_colorize(text_art)

# Print result
print(colored_art)
```

### Transformation Methods

```python
from figlet_forge import Figlet

# Create a Figlet instance
fig = Figlet(font="standard")

# Render text
text_art = fig.renderText("Transform")

# Apply transformations
reversed_art = text_art.reverse()
flipped_art = text_art.flip()
bordered_art = text_art.border()

# Print results
print("Reversed:")
print(reversed_art)
print("\nFlipped:")
print(flipped_art)
print("\nBordered:")
print(bordered_art)
```

## Integration Examples

### Terminal Welcome Message

Create a custom terminal welcome message:

```python
from figlet_forge import Figlet
from figlet_forge.color.effects import gradient_colorize

def create_welcome_message(username):
    fig = Figlet(font="big", width=80, justify="center")
    welcome = fig.renderText(f"Welcome, {username}!")
    colored = gradient_colorize(welcome, "BLUE", "CYAN")
    return colored

# Save to ~/.bashrc or similar
with open("welcome.txt", "w") as f:
    f.write(create_welcome_message("User"))
```

## Troubleshooting

### Font Not Found

If you encounter a "Font not found" error, check:

1. The font name spelling is correct
2. The font is installed (use `--list-fonts` to see available fonts)
3. Try installing the font with `--install-font`

### Color Not Displaying

If colors aren't displaying:

1. Check your terminal supports ANSI colors
2. Try using a different color or effect
3. Use the `--no-color` flag if your terminal doesn't support colors

## Conclusion

Figlet Forge provides a powerful way to create beautiful ASCII text art with advanced features like colors, transformations, and Unicode support. Whether you're spicing up CLI tools, creating welcome banners, or just having fun with text, Figlet Forge offers elegant, precise typography following Eidosian principles.

For more examples, see the `examples/` directory in the project repository or visit our documentation site.

> _"Elegance in recursion, precision in expression."_ - Lloyd Handyside
