#!/usr/bin/env python3
"""
Colors

Description of the module's purpose and functionality following Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction

This module should embody core Terminal Forge architectural patterns while maintaining
crystal-clear intent and optimal performance characteristics.

🌈 Eidosian Color System 🎨
A precision-engineered terminal color toolkit balancing expressiveness with efficiency.

Quantum-level color control with zero wasted electrons.
"""
import os
import sys
import re
from typing import Tuple, Dict, Union, Optional
from functools import lru_cache

class Color:
    """
    Terminal color system with automatic capability detection and optimization.
    Each color exists only when observed—quantum efficiency in action.
    """
    # ╔═══════════════════╗
    # ║ ANSI Color Codes  ║
    # ╚═══════════════════╝
    RESET = "\033[0m"
    # Base colors - the fundamental palette
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright variants - the expanded spectrum
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    # Styles - the dimensional enhancers
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    
    # Color collections for systematic access
    COLORS = {
        "black": BLACK,
        "red": RED,
        "green": GREEN,
        "yellow": YELLOW,
        "blue": BLUE,
        "magenta": MAGENTA,
        "cyan": CYAN,
        "white": WHITE,
        "bright_black": BRIGHT_BLACK,
        "bright_red": BRIGHT_RED,
        "bright_green": BRIGHT_GREEN,
        "bright_yellow": BRIGHT_YELLOW,
        "bright_blue": BRIGHT_BLUE,
        "bright_magenta": BRIGHT_MAGENTA,
        "bright_cyan": BRIGHT_CYAN,
        "bright_white": BRIGHT_WHITE,
    }

    @staticmethod
    @lru_cache(maxsize=32)
    def rgb(r: int, g: int, b: int) -> str:
        """
        Generate RGB color code with memoization for quantum-computing-level efficiency.
        Each unique color computes only once, then exists forever in the cache dimension.
        """
        # Parameter validation with humor
        if not all(0 <= x <= 255 for x in (r, g, b)):
            raise ValueError(f"RGB values ({r},{g},{b}) must be 0-255. Colors exist in a finite universe! 🌌")
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    @lru_cache(maxsize=32)
    def bg_rgb(r: int, g: int, b: int) -> str:
        """
        Generate RGB background color with ultimate performance optimization.
        Why compute twice what can be computed once and cached into infinity?
        """
        if not all(0 <= x <= 255 for x in (r, g, b)):
            raise ValueError(f"RGB values ({r},{g},{b}) must be 0-255. Even backgrounds have standards! 🖼️")
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def rainbow(text: str) -> str:
        """
        Transform text into the full spectrum of visible light.
        As cosmic as a supernova, as precise as quantum mechanics.
        """
        if not text:
            return ""  # Empty string? Empty rainbow. Conservation of energy.
            
        colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.CYAN, Color.BLUE, Color.MAGENTA]
        result = ""
        for i, char in enumerate(text):
            if char.strip():  # Color only visible matter, not spaces
                result += f"{colors[i % len(colors)]}{char}{Color.RESET}"
            else:
                result += char
        return result

    @staticmethod
    def gradient(text: str, start_color: Tuple[int, int, int], end_color: Tuple[int, int, int]) -> str:
        """
        Apply a mathematically perfect gradient transition to text.
        The spectrum flows like thoughts through a quantum neural network.
        """
        if not text:
            return ""
        
        result = ""
        for i, char in enumerate(text):
            if char.strip():  # Empty space carries no color
                # Linear interpolation with single-pass efficiency
                progress = i / max(len(text)-1, 1)  # Avoid division by zero—universe intact
                r = int(start_color[0] + (end_color[0] - start_color[0]) * progress)
                g = int(start_color[1] + (end_color[1] - start_color[1]) * progress)
                b = int(start_color[2] + (end_color[2] - start_color[2]) * progress)
                result += f"{Color.rgb(r, g, b)}{char}{Color.RESET}"
            else:
                result += char
        return result
    
    @staticmethod
    def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
        """
        Blend two ANSI colors with mathematical precision.
        Like mixing quantum states, but for your terminal.
        """
        # Extract RGB values with regex precision
        pattern = r'\033\[38;2;(\d+);(\d+);(\d+)m'
        match1 = re.search(pattern, color1)
        match2 = re.search(pattern, color2)
        
        if not (match1 and match2):
            raise ValueError("Can only blend RGB colors, not the fabric of spacetime! 🌠")
            
        # Extract and blend with perfect mathematical harmony
        r1, g1, b1 = map(int, match1.groups())
        r2, g2, b2 = map(int, match2.groups())
        
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        
        return Color.rgb(r, g, b)
        
    @staticmethod
    def supports_color() -> bool:
        """
        Detect terminal's color capability with quantum certainty.
        Your terminal either supports color or it doesn't—no superposition allowed.
        """
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            ('COLORTERM' in os.environ or 
             ('TERM' in os.environ and os.environ.get('TERM') != 'dumb'))
        )
    
    @staticmethod
    def get_color(name: str) -> str:
        """
        Retrieve color by name with fail-safe handling.
        Even in the multiverse, we ensure you get the right color or know why.
        """
        name = name.lower()
        if name in Color.COLORS:
            return Color.COLORS[name]
        
        # Handle hex format
        if name.startswith('#') and len(name) == 7:
            try:
                r = int(name[1:3], 16)
                g = int(name[3:5], 16)
                b = int(name[5:7], 16)
                return Color.rgb(r, g, b)
            except ValueError:
                pass
        
        # When all else fails, return safely with quantum humor
        raise ValueError(f"Color '{name}' not found in this universe's spectrum! 🔭")

# Self-testing capability demonstrates perfect self-awareness
if __name__ == "__main__":
    print(f"{Color.BOLD}{Color.RED}Red and bold!{Color.RESET}")
    print(f"Rainbow: {Color.rainbow('Experience the quantum spectrum!')}")
    print(f"Gradient: {Color.gradient('Smooth color transition', (255,0,0), (0,0,255))}")


