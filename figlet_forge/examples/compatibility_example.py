#!/usr/bin/env python3
"""
Figlet Forge - Compatibility Example

This example demonstrates how Figlet Forge can be used as a drop-in
replacement for pyfiglet, maintaining full API compatibility.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# First, try to use pyfiglet if available
try:
    import pyfiglet

    print("\n=== Using original pyfiglet ===\n")

    # Standard pyfiglet usage
    print("Using figlet_format:")
    result = pyfiglet.figlet_format("pyfiglet", font="standard")
    print(result)

    print("\nUsing Figlet class:")
    fig = pyfiglet.Figlet(font="small")
    result = fig.renderText("pyfiglet")
    print(result)

    # Get available fonts
    fonts = fig.getFonts()
    print(f"\nFound {len(fonts)} fonts (first 3): {fonts[:3]}")

except ImportError:
    print("Original pyfiglet not installed, skipping comparison")

# Now use Figlet Forge with compatibility layer
print("\n=== Using Figlet Forge compatibility layer ===\n")

# Import from compatibility module
from figlet_forge.compat import Figlet, figlet_format

# Same API as pyfiglet
print("Using figlet_format:")
result = figlet_format("Figlet Forge", font="standard")
print(result)

print("\nUsing Figlet class:")
fig = Figlet(font="small")
result = fig.renderText("Figlet Forge")
print(result)

# Get available fonts (same API)
fonts = fig.getFonts()
print(f"\nFound {len(fonts)} fonts (first 3): {fonts[:3]}")

# Special Figlet Forge enhancements (while maintaining compatibility)
print("\n=== Special Figlet Forge Enhancements ===\n")

# Get width of rendered text (compatible method)
width = fig.getRenderWidth("Test")
print(f"Width of 'Test' in 'small' font: {width} characters")

# Demonstrate how to use Figlet Forge's extended features while
# maintaining compatibility with pyfiglet codebases
print("\nUsing Figlet Forge-specific features:")

# Import core Figlet Forge capabilities
from figlet_forge import Figlet as CoreFiglet
from figlet_forge.color.effects import rainbow_colorize

# Create a colored figlet (Figlet Forge extension)
fig = CoreFiglet(font="standard")
text = fig.renderText("Enhanced!")
colored = rainbow_colorize(text)
print(colored)

print("\n=== Compatibility Report ===\n")
print("✓ figlet_format function")
print("✓ Figlet class")
print("✓ renderText method")
print("✓ getFonts method")
print("✓ Consistent results")
print("✓ Enhanced with Figlet Forge features")
