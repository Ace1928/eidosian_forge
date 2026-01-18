"""
ASCII Art - Visual Transformation Engine 🖼️

This module transforms visual elements into terminal-compatible ASCII/Unicode art
with mathematical precision and contextual awareness.

Core capabilities:
- Image-to-ASCII/Unicode conversion with perceptual mapping
- Unicode block intelligence for higher fidelity rendering
- Terminal-aware color application with dynamic adaptation
- Width-aware scaling with precise aspect preservation

Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction
- Structure as Control: Clear architectural boundaries with natural composition
"""

import os
from typing import Literal, Optional, Tuple, Union, Dict, List, Any, Callable, TypeVar, Protocol
from pathlib import Path
import math
from functools import lru_cache

try:
    from PIL import Image, ImageOps, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ╭──────────────────────────────────────────────────────╮
# │  🧬 Types & Protocols - Structural DNA               │
# ╰──────────────────────────────────────────────────────╯

T = TypeVar('T')
ImageSource = Union[str, Path, 'Image.Image', bytes]

class PixelMatrix(Protocol):
    """Protocol for any object that behaves like a pixel matrix."""
    
    def __getitem__(self, xy: Tuple[int, int]) -> Union[Tuple[int, ...], int]: ...

# ╭──────────────────────────────────────────────────────╮
# │  🎨 Character Sets - Alphabets of Visual Alchemy     │
# ╰──────────────────────────────────────────────────────╯

# Standard sets with varied density and styling characteristics
CHAR_SETS = {
    # Classic ASCII sets - low to high density
    "minimal": " .:;+=xX$&@",
    "standard": " .,:;+*?%S#@",
    "extended": " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "inverted": "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    
    # Unicode block sets - higher resolution rendering
    "blocks": " ▏▎▍▌▋▊▉█",  # 1/8th width blocks for smooth gradients
    "shades": " ░▒▓█",      # Shade blocks for distinct steps
    "quadrants": "  ▖▗▘▙▚▛▜▝▞▟█", # 2x2 block elements for higher resolution
    
    # Specialized sets for specific use cases
    "braille": " ⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿",
    "technical": " ○•◘■□▪▫▬▲►▼◄△▴▵▶▷▸▹▻▽▾▿◁◃◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯",
    "weather": "☀☁☂☃☉☼☾☽♁♨❄❅❆✵✶✷✸✹✺❇❈❉❊",
}

# Perceptually calibrated character sets - scientifically balanced for visual density
PERCEPTUAL_SETS = {
    "ascii_clean": " .,:;irsXA253hMHGS#9B&@", # Visually balanced ASCII gradient
    "ascii_fine": " .'`^\",:;Il!i><~+_-?][}{1)(|\\tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$", # Fine detail ASCII
    "blocks_balanced": " ▁▂▃▄▅▆▇█",           # Height-balanced blocks
    "dots": " ░▒▓█",                          # Pure dot density progression
    "ultra_smooth": " ▏▎▍▌▋▊▉█",              # Extremely smooth transition
}

# Sets optimized for specific use cases
SPECIALIZED_SETS = {
    "matrix": ".:-=+*#%@",                   # Classic "Matrix" style
    "sketch": " .'`^\",:;Il!i><~+_-?][}{1)(|", # Pen sketch appearance
    "shadow": " ░▒▓█",                        # Shadow/darkness representation
    "lines": "╵╷│┃║╹╻╽╿╏┇┋┊┆┼",               # Line-drawing characters 
    "box_drawing": "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋"
}

# Combined set for convenient access
ALL_CHAR_SETS = {**CHAR_SETS, **PERCEPTUAL_SETS, **SPECIALIZED_SETS}

# ╭──────────────────────────────────────────────────────╮
# │  🧮 Dithering Algorithms - Error Diffusion Mastery   │
# ╰──────────────────────────────────────────────────────╯

from typing import Type, Protocol, runtime_checkable
import array

@runtime_checkable
class DitheringAlgorithm(Protocol):
    """Protocol defining dithering algorithm interface with guaranteed behavior."""
    
    name: str
    description: str
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply dithering to an image.
        
        Args:
            image: PIL Image to dither
            
        Returns:
            Dithered PIL Image
        """
        ...


class FloydSteinberg:
    """Floyd-Steinberg dithering - classic error diffusion with 16-unit pattern.
    
    Implements the original Floyd-Steinberg algorithm (1976) with the standard
    diffusion pattern that spreads error to neighboring pixels:
                     *  7/16
            3/16  5/16  1/16
    
    Known for good quality and moderate computational efficiency.
    """
    
    name = "floyd-steinberg"
    description = "Classic error diffusion (7/16, 3/16, 5/16, 1/16 pattern)"
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply Floyd-Steinberg dithering using optimized implementation."""
        # Convert to grayscale for consistent processing
        img = image.convert('L')
        width, height = img.size
        
        # Create array for faster manipulation
        pixels = array.array('B', img.tobytes())
        threshold = 128  # Mid-gray threshold for binary decision
        
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                old_pixel = pixels[idx]
                new_pixel = 255 if old_pixel >= threshold else 0
                pixels[idx] = new_pixel
                
                # Calculate quantization error
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels according to F-S pattern
                if x < width - 1:
                    # Right pixel (7/16)
                    pixels[idx + 1] = min(255, max(0, pixels[idx + 1] + (error * 7) // 16))
                
                if y < height - 1:
                    next_row = idx + width
                    
                    if x > 0:
                        # Bottom-left pixel (3/16)
                        pixels[next_row - 1] = min(255, max(0, pixels[next_row - 1] + (error * 3) // 16))
                    
                    # Bottom pixel (5/16)
                    pixels[next_row] = min(255, max(0, pixels[next_row] + (error * 5) // 16))
                    
                    if x < width - 1:
                        # Bottom-right pixel (1/16)
                        pixels[next_row + 1] = min(255, max(0, pixels[next_row + 1] + (error * 1) // 16))
        
        # Create new image from the processed pixels
        return Image.frombytes('L', (width, height), pixels.tobytes())


class Atkinson:
    """Atkinson dithering - balanced quality with 1/8 error distribution.
    
    Popularized by Bill Atkinson for Apple's early Macintosh graphics. Distributes
    only 3/4 of the error to 6 neighboring pixels in a unique pattern:
                    *  1/8  1/8
            1/8  1/8  1/8
                    1/8
    
    Produces clean, slightly higher contrast results than Floyd-Steinberg.
    """
    
    name = "atkinson"
    description = "Balanced 3/4 error distribution (Apple style)"
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply Atkinson dithering with optimized error diffusion."""
        img = image.convert('L')
        width, height = img.size
        
        # Create working copy as array for faster manipulation
        pixels = array.array('B', img.tobytes())
        threshold = 128  # Mid-gray threshold
        
        # Process pixels efficiently in a single pass
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                old_pixel = pixels[idx]
                new_pixel = 255 if old_pixel >= threshold else 0
                pixels[idx] = new_pixel
                
                # Calculate error (Atkinson only distributes 3/4 of the error)
                error = (old_pixel - new_pixel) >> 3  # Divide by 8 efficiently
                
                # Atkinson pattern - 1/8 to each of 6 pixels
                # Right and far right
                if x + 1 < width:
                    pixels[idx + 1] = min(255, max(0, pixels[idx + 1] + error))
                if x + 2 < width:
                    pixels[idx + 2] = min(255, max(0, pixels[idx + 2] + error))
                
                # Next row: left, center, right
                if y + 1 < height:
                    row_offset = idx + width
                    if x > 0:
                        pixels[row_offset - 1] = min(255, max(0, pixels[row_offset - 1] + error))
                    pixels[row_offset] = min(255, max(0, pixels[row_offset] + error))
                    if x + 1 < width:
                        pixels[row_offset + 1] = min(255, max(0, pixels[row_offset + 1] + error))
                
                # Two rows down, center
                if y + 2 < height:
                    pixels[idx + 2 * width] = min(255, max(0, pixels[idx + 2 * width] + error))
        
        # Create new image with dithered array
        return Image.frombytes('L', (width, height), pixels.tobytes())


class JarvisJudiceNinke:
    """Jarvis-Judice-Ninke dithering - wide error diffusion for superior quality.
    
    Published in 1976, this algorithm distributes error to 12 neighboring pixels
    across three rows using a 48-unit denominator. Known for excellent quality
    at the cost of higher computational demand.
    
    Error distribution pattern:
                      *  7  5
        3  5  7  5  3
        1  3  5  3  1
    """
    
    name = "jarvis-judice-ninke"
    description = "Wide error diffusion (12 neighboring pixels, 48-unit pattern)"
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply JJN dithering with precise error diffusion matrix."""
        img = image.convert('L')
        width, height = img.size
        
        # Create working copy as array for faster manipulation
        pixels = array.array('B', img.tobytes())
        threshold = 128
        
        # Process pixels with full JJN pattern
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                old_pixel = pixels[idx]
                new_pixel = 255 if old_pixel >= threshold else 0
                pixels[idx] = new_pixel
                
                # Calculate error
                error = old_pixel - new_pixel
                
                # Current row (forward only)
                if x + 1 < width:
                    pixels[idx + 1] = min(255, max(0, pixels[idx + 1] + error * 7 // 48))
                if x + 2 < width:
                    pixels[idx + 2] = min(255, max(0, pixels[idx + 2] + error * 5 // 48))
                
                # Next row - five positions
                if y + 1 < height:
                    next_row_idx = idx + width
                    for dx, weight in [(-2, 3), (-1, 5), (0, 7), (1, 5), (2, 3)]:
                        nx = x + dx
                        if 0 <= nx < width:
                            pixels[next_row_idx + dx] = min(255, max(0, 
                                pixels[next_row_idx + dx] + error * weight // 48))
                
                # Two rows down - five positions
                if y + 2 < height:
                    next_row_idx = idx + width * 2
                    for dx, weight in [(-2, 1), (-1, 3), (0, 5), (1, 3), (2, 1)]:
                        nx = x + dx
                        if 0 <= nx < width:
                            pixels[next_row_idx + dx] = min(255, max(0, 
                                pixels[next_row_idx + dx] + error * weight // 48))
        
        return Image.frombytes('L', (width, height), pixels.tobytes())


class Sierra:
    """Sierra dithering - balanced distribution with reduced bleed.
    
    Sierra's algorithm (1989) is a variant of JJN with a modified pattern that
    provides excellent quality with slightly less blurring. Uses a 32-unit
    distribution across 3 rows.
    
    Error distribution pattern:
                      *  5  3
        2  4  5  4  2
           2  3  2
    """
    
    name = "sierra"
    description = "Sierra error distribution (reduced bleed, 32-unit pattern)"
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply Sierra dithering with optimized implementation."""
        img = image.convert('L')
        width, height = img.size
        
        pixels = array.array('B', img.tobytes())
        threshold = 128
        
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                old_pixel = pixels[idx]
                new_pixel = 255 if old_pixel >= threshold else 0
                pixels[idx] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Current row - forward only
                if x + 1 < width:
                    pixels[idx + 1] = min(255, max(0, pixels[idx + 1] + error * 5 // 32))
                if x + 2 < width:
                    pixels[idx + 2] = min(255, max(0, pixels[idx + 2] + error * 3 // 32))
                
                # Next row - five positions
                if y + 1 < height:
                    next_row_idx = idx + width
                    for dx, weight in [(-2, 2), (-1, 4), (0, 5), (1, 4), (2, 2)]:
                        nx = x + dx
                        if 0 <= nx < width:
                            pixels[next_row_idx + dx] = min(255, max(0, 
                                pixels[next_row_idx + dx] + error * weight // 32))
                
                # Two rows down - three positions
                if y + 2 < height:
                    next_row_idx = idx + width * 2
                    for dx, weight in [(-1, 2), (0, 3), (1, 2)]:
                        nx = x + dx
                        if 0 <= nx < width:
                            pixels[next_row_idx + dx] = min(255, max(0, 
                                pixels[next_row_idx + dx] + error * weight // 32))
        
        return Image.frombytes('L', (width, height), pixels.tobytes())


class StuckiDithering:
    """Stucki dithering - enhanced JJN with optimized error diffusion.
    
    Developed by Peter Stucki (1981), this algorithm further refines JJN with
    a modified error distribution pattern that produces excellent detail and
    edge preservation using a 42-unit denominator.
    
    Error distribution pattern:
                      *  8  4
        2  4  8  4  2
        1  2  4  2  1
    """
    
    name = "stucki"
    description = "Optimized JJN variant with enhanced edge detail"
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply Stucki dithering algorithm with optimized implementation."""
        img = image.convert('L')
        width, height = img.size
        
        pixels = array.array('B', img.tobytes())
        threshold = 128
        
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                old_pixel = pixels[idx]
                new_pixel = 255 if old_pixel >= threshold else 0
                pixels[idx] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Current row - forward only
                if x + 1 < width:
                    pixels[idx + 1] = min(255, max(0, pixels[idx + 1] + error * 8 // 42))
                if x + 2 < width:
                    pixels[idx + 2] = min(255, max(0, pixels[idx + 2] + error * 4 // 42))
                
                # Next row - five positions
                if y + 1 < height:
                    next_row_idx = idx + width
                    for dx, weight in [(-2, 2), (-1, 4), (0, 8), (1, 4), (2, 2)]:
                        nx = x + dx
                        if 0 <= nx < width:
                            pixels[next_row_idx + dx] = min(255, max(0, 
                                pixels[next_row_idx + dx] + error * weight // 42))
                
                # Two rows down - five positions
                if y + 2 < height:
                    next_row_idx = idx + width * 2
                    for dx, weight in [(-2, 1), (-1, 2), (0, 4), (1, 2), (2, 1)]:
                        nx = x + dx
                        if 0 <= nx < width:
                            pixels[next_row_idx + dx] = min(255, max(0, 
                                pixels[next_row_idx + dx] + error * weight // 42))
        
        return Image.frombytes('L', (width, height), pixels.tobytes())


class NoDither:
    """Simple threshold-based conversion with no error diffusion.
    
    Applies a pure threshold operation without any error diffusion,
    resulting in high-contrast binary output with sharp edges and
    no halftone simulation.
    """
    
    name = "none"
    description = "Pure threshold conversion without error diffusion"
    
    @staticmethod
    def apply(image: 'Image.Image') -> 'Image.Image':
        """Apply simple thresholding without dithering."""
        return image.convert('L').point(lambda x: 255 if x >= 128 else 0, '1')


class DitheringRegistry:
    """Registry of available dithering algorithms for dynamic discovery and management."""
    
    _algorithms: Dict[str, Type] = {
        "none": NoDither,
        "floyd-steinberg": FloydSteinberg,
        "atkinson": Atkinson,
        "jarvis-judice-ninke": JarvisJudiceNinke,
        "sierra": Sierra,
        "stucki": StuckiDithering,
    }
    
    @classmethod
    def get(cls, name: str) -> Type:
        """Get dithering algorithm implementation by name.
        
        Args:
            name: Algorithm identifier
            
        Returns:
            Algorithm class implementation
            
        Raises:
            ValueError: If algorithm name is not recognized
        """
        if name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(f"Unknown dithering algorithm '{name}'. Available: {available}")
        return cls._algorithms[name]
    
    @classmethod
    def register(cls, algorithm: Type) -> None:
        """Register a new dithering algorithm.
        
        Args:
            algorithm: Algorithm class to register (must have name attribute and apply method)
            
        Raises:
            ValueError: If algorithm doesn't implement the required interface
        """
        if not hasattr(algorithm, 'name') or not hasattr(algorithm, 'apply'):
            raise ValueError("Algorithm must have 'name' attribute and 'apply' method")
        cls._algorithms[algorithm.name] = algorithm
    
    @classmethod
    def list_algorithms(cls) -> Dict[str, str]:
        """Get all available algorithms with descriptions.
        
        Returns:
            Dictionary mapping algorithm names to descriptions
        """
        return {name: algo.description for name, algo in cls._algorithms.items()}

# Replace the old DITHERING_ALGORITHMS dict with the registry
DITHERING_ALGORITHMS = DitheringRegistry._algorithms

# ╭──────────────────────────────────────────────────────╮
# │  🌈 Color Mapping - Perceptual Intelligence System   │
# ╰──────────────────────────────────────────────────────╯

class ColorSystem:
    """Color processing system with perceptual mapping for terminal environments.
    
    Provides intelligent color space transformations optimized for human visual perception
    and terminal color capabilities. Maps RGB values to appropriate terminal color codes
    while maintaining maximum visual fidelity within the constraints of each terminal mode.
    """
    
    # ANSI color definitions - standard color palette for terminals
    # Indexed by color name for semantic access
    ANSI_COLORS = {
        # Standard colors (30-37)
        "black": (0, 0, 0), 
        "red": (170, 0, 0), 
        "green": (0, 170, 0), 
        "yellow": (170, 85, 0),
        "blue": (0, 0, 170), 
        "magenta": (170, 0, 170), 
        "cyan": (0, 170, 170), 
        "white": (170, 170, 170),
        
        # Bright variants (90-97)
        "bright_black": (85, 85, 85),
        "bright_red": (255, 85, 85),
        "bright_green": (85, 255, 85),
        "bright_yellow": (255, 255, 85),
        "bright_blue": (85, 85, 255),
        "bright_magenta": (255, 85, 255),
        "bright_cyan": (85, 255, 255),
        "bright_white": (255, 255, 255)
    }
    
    # Color code lookup for ANSI terminal modes
    ANSI_CODES = {
        # Foreground codes
        "fg": {
            "black": 30, "red": 31, "green": 32, "yellow": 33,
            "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
            "bright_black": 90, "bright_red": 91, "bright_green": 92, 
            "bright_yellow": 93, "bright_blue": 94, "bright_magenta": 95,
            "bright_cyan": 96, "bright_white": 97
        },
        # Background codes
        "bg": {
            "black": 40, "red": 41, "green": 42, "yellow": 43,
            "blue": 44, "magenta": 45, "cyan": 46, "white": 47,
            "bright_black": 100, "bright_red": 101, "bright_green": 102,
            "bright_yellow": 103, "bright_blue": 104, "bright_magenta": 105,
            "bright_cyan": 106, "bright_white": 107
        }
    }
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB color to CIE-LAB color space for perceptual operations.
        
        CIE-LAB is designed to approximate human vision and provides better perceptual
        uniformity than RGB or HSL spaces. This enables more accurate color comparisons.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            Tuple of (L*, a*, b*) values in CIE-LAB color space
        """
        # Normalize RGB to [0, 1]
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        
        # Apply gamma correction (sRGB to linear RGB)
        r = (r / 12.92) if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g = (g / 12.92) if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b = (b / 12.92) if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        
        # Linear RGB to XYZ using D65 white point matrix
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # XYZ to LAB
        # Reference D65 white point
        x_n, y_n, z_n = 0.95047, 1.0, 1.08883
        
        # Normalize XYZ values
        x, y, z = x / x_n, y / y_n, z / z_n
        
        # Apply nonlinear transformation
        x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
        y = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
        z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)
        
        L = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        return (L, a, b)
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate perceptual distance between two colors using CIE94 formula.
        
        Uses a more advanced formula than simple Euclidean distance to better
        match human perception of color differences.
        
        Args:
            color1: First RGB color tuple (r, g, b)
            color2: Second RGB color tuple (r, g, b)
            
        Returns:
            Perceptual distance value (smaller = more similar)
        """
        # Convert both colors to LAB space
        L1, a1, b1 = ColorSystem.rgb_to_lab(*color1)
        L2, a2, b2 = ColorSystem.rgb_to_lab(*color2)
        
        # Calculate CIE94 color difference
        # Constants for graphic arts
        k_L, k_1, k_2 = 1, 0.045, 0.015
        
        # Calculate deltas
        delta_L = L1 - L2
        delta_a = a1 - a2
        delta_b = b1 - b2
        
        # Calculate chroma and hue differences
        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        delta_C = C1 - C2
        
        delta_H_squared = delta_a**2 + delta_b**2 - delta_C**2
        # Ensure non-negative value due to potential floating point errors
        delta_H = math.sqrt(max(0, delta_H_squared))
        
        # Calculate the weighted components
        S_L = 1  # Always 1 for graphic arts
        S_C = 1 + k_1 * C1
        S_H = 1 + k_2 * C1
        
        # Final CIE94 distance formula
        delta_E94 = math.sqrt(
            (delta_L / (k_L * S_L))**2 +
            (delta_C / S_C)**2 +
            (delta_H / S_H)**2
        )
        
        return delta_E94
    
    @staticmethod
    @lru_cache(maxsize=512)
    def find_closest_ansi_color(r: int, g: int, b: int, mode: int = 8) -> Tuple[int, str]:
        """Find the closest ANSI color to a given RGB value.
        
        Uses perceptual color matching to find the best terminal color representation.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            mode: Terminal color mode (8 or 16)
            
        Returns:
            Tuple of (ANSI code, color name)
        """
        # Determine which colors are available based on mode
        colors = list(ColorSystem.ANSI_COLORS.items())
        if mode == 8:  # 8-color mode, only use the base colors
            colors = [(name, rgb) for name, rgb in colors if not name.startswith('bright_')]
            
        # Find closest color using perceptual color distance
        min_distance = float('inf')
        closest_name = 'white'  # Default fallback
        
        for name, color_rgb in colors:
            distance = ColorSystem.color_distance((r, g, b), color_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        # Get the corresponding ANSI code
        ansi_code = ColorSystem.ANSI_CODES['fg'][closest_name]
        
        return (ansi_code, closest_name)

    @staticmethod
    @lru_cache(maxsize=512)
    def rgb_to_256color_index(r: int, g: int, b: int) -> int:
        """Convert RGB to an index in the 256-color palette.
        
        Maps RGB values to the terminal's 256-color palette using the standard
        color cube mapping method (6×6×6 color cube + 24 grayscale levels).
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            Index in 256-color palette (16-255)
        """
        # Check if this is grayscale (R=G=B)
        if r == g == b:
            # Map to grayscale range (232-255, 24 levels)
            # Calculate closest level from the 24 available gray shades
            gray_idx = min(23, max(0, round((r / 255) * 23)))
            return 232 + gray_idx
        
        # Map to 6×6×6 RGB color cube (indices 16-231)
        # Each RGB component is quantized to 6 levels: 0, 51, 102, 153, 204, 255
        r_idx = min(5, max(0, round((r / 255) * 5)))
        g_idx = min(5, max(0, round((g / 255) * 5)))
        b_idx = min(5, max(0, round((b / 255) * 5)))
        
        # Calculate cube index: 16 + 36*r + 6*g + b
        return 16 + (r_idx * 36) + (g_idx * 6) + b_idx
    
    @staticmethod
    def apply_color(
        char: str, 
        fg_color: Tuple[int, int, int],
        bg_color: Optional[Tuple[int, int, int]] = None,
        color_mode: int = 8
    ) -> str:
        """Apply terminal color to a character with optimal encoding for the terminal mode.
        
        Intelligently maps colors to the appropriate terminal color capability,
        handling different color depths from monochrome to 24-bit TrueColor.
        
        Args:
            char: Character to colorize
            fg_color: RGB foreground color tuple (r, g, b)
            bg_color: Optional RGB background color tuple (r, g, b)
            color_mode: Terminal color depth (1, 8, 16, 256, or 16M)
            
        Returns:
            ANSI color-coded string
        """
        if color_mode == 1:
            # Monochrome mode - no color support
            return char
        
        r, g, b = fg_color
        
        # Start building ANSI escape sequence
        fg_code = ""
        bg_code = ""
        
        # Process foreground color based on terminal capabilities
        if color_mode <= 16:
            # 8 or 16 color terminal - map to basic ANSI colors
            ansi_code, _ = ColorSystem.find_closest_ansi_color(r, g, b, color_mode)
            fg_code = f"\033[{ansi_code}m"
        
        elif color_mode == 256:
            # 256-color terminal - use extended color codes
            color_idx = ColorSystem.rgb_to_256color_index(r, g, b)
            fg_code = f"\033[38;5;{color_idx}m"
        
        else:
            # True color (16M) terminal - use direct RGB values
            fg_code = f"\033[38;2;{r};{g};{b}m"
        
        # Process background color if provided
        if bg_color:
            bg_r, bg_g, bg_b = bg_color
            
            if color_mode <= 16:
                # 8 or 16 color terminal
                bg_ansi_code, _ = ColorSystem.find_closest_ansi_color(bg_r, bg_g, bg_b, color_mode)
                # Convert foreground code to background (40-47 or 100-107)
                bg_index = ColorSystem.ANSI_CODES['bg'][_]
                bg_code = f"\033[{bg_index}m"
            
            elif color_mode == 256:
                # 256-color terminal
                bg_color_idx = ColorSystem.rgb_to_256color_index(bg_r, bg_g, bg_b)
                bg_code = f"\033[48;5;{bg_color_idx}m"
            
            else:
                # True color terminal
                bg_code = f"\033[48;2;{bg_r};{bg_g};{bg_b}m"
        
        # Combine codes, add character, and reset
        return f"{fg_code}{bg_code}{char}\033[0m"

# Replace the existing function with ColorSystem's optimized version
def apply_terminal_color(
    char: str, 
    r: int, 
    g: int, 
    b: int, 
    color_mode: int = 8,
    bg_color: Optional[Tuple[int, int, int]] = None
) -> str:
    """Apply terminal color to a character based on RGB values.
    
    Args:
        char: Character to colorize
        r, g, b: RGB color components (0-255)
        color_mode: Terminal color depth (1, 8, 16, 256, 16M)
        bg_color: Optional background color as RGB tuple
        
    Returns:
        ANSI color-coded string
    """
    return ColorSystem.apply_color(char, (r, g, b), bg_color, color_mode)


class ColorDithering:
    """Color dithering algorithms for reducing color banding in limited color environments.
    
    Implements error diffusion algorithms specifically designed for color reduction,
    allowing for better color representation in terminals with limited color capability.
    """
    
    @staticmethod
    def floyd_steinberg(
        image: 'Image.Image', 
        palette_size: int = 16
    ) -> 'Image.Image':
        """Apply Floyd-Steinberg dithering for color reduction.
        
        Preserves perceptual color fidelity while reducing to a smaller palette.
        
        Args:
            image: PIL Image to dither
            palette_size: Target color palette size (8, 16, or 256)
            
        Returns:
            Dithered PIL Image with reduced colors
        """
        # Convert to RGB to ensure we're working with color
        img = image.convert('RGB')
        width, height = img.size
        
        # Create pixel access objects
        pixels = img.load()
        
        # Function to quantize a color to target palette
        def quantize_color(r: int, g: int, b: int) -> Tuple[int, int, int]:
            if palette_size == 8 or palette_size == 16:
                # Map to ANSI colors
                _, color_name = ColorSystem.find_closest_ansi_color(r, g, b, palette_size)
                return ColorSystem.ANSI_COLORS[color_name]
            else:
                # 256 colors - use the 6×6×6 color cube quantization
                steps = [0, 51, 102, 153, 204, 255]
                r_q = steps[min(5, max(0, round((r / 255) * 5)))]
                g_q = steps[min(5, max(0, round((g / 255) * 5)))]
                b_q = steps[min(5, max(0, round((b / 255) * 5)))]
                return (r_q, g_q, b_q)
        
        # Process the image with Floyd-Steinberg error diffusion
        for y in range(height):
            for x in range(width):
                # Get original pixel color
                r, g, b = pixels[x, y]
                
                # Quantize to target palette
                r_new, g_new, b_new = quantize_color(r, g, b)
                
                # Set the new pixel value
                pixels[x, y] = (r_new, g_new, b_new)
                
                # Calculate quantization error
                error_r = r - r_new
                error_g = g - g_new
                error_b = b - b_new
                
                # Distribute errors - Floyd-Steinberg pattern
                # Right pixel (7/16)
                if x < width - 1:
                    r_right, g_right, b_right = pixels[x + 1, y]
                    pixels[x + 1, y] = (
                        min(255, max(0, r_right + (error_r * 7) // 16)),
                        min(255, max(0, g_right + (error_g * 7) // 16)),
                        min(255, max(0, b_right + (error_b * 7) // 16))
                    )
                
                # Next row pixels
                if y < height - 1:
                    # Bottom-left pixel (3/16)
                    if x > 0:
                        r_bl, g_bl, b_bl = pixels[x - 1, y + 1]
                        pixels[x - 1, y + 1] = (
                            min(255, max(0, r_bl + (error_r * 3) // 16)),
                            min(255, max(0, g_bl + (error_g * 3) // 16)),
                            min(255, max(0, b_bl + (error_b * 3) // 16))
                        )
                    
                    # Bottom pixel (5/16)
                    r_b, g_b, b_b = pixels[x, y + 1]
                    pixels[x, y + 1] = (
                        min(255, max(0, r_b + (error_r * 5) // 16)),
                        min(255, max(0, g_b + (error_g * 5) // 16)),
                        min(255, max(0, b_b + (error_b * 5) // 16))
                    )
                    
                    # Bottom-right pixel (1/16)
                    if x < width - 1:
                        r_br, g_br, b_br = pixels[x + 1, y + 1]
                        pixels[x + 1, y + 1] = (
                            min(255, max(0, r_br + (error_r * 1) // 16)),
                            min(255, max(0, g_br + (error_g * 1) // 16)),
                            min(255, max(0, b_br + (error_b * 1) // 16))
                        )
        
        return img

# ╭──────────────────────────────────────────────────────╮
# │  🖼️ Core Transformation Engine - Pixel Alchemy      │
# ╰──────────────────────────────────────────────────────╯

def convert(
    image_path: Union[str, Path],
    style: Literal["minimal", "detailed", "artistic"] = "detailed",
    character_set: Literal["minimal", "standard", "extended", "inverted", "blocks", "braille", "shades"] = "standard",
    width: int = 80,
    height: Optional[int] = None,
    color_mode: int = 8,
    dithering: str = "floyd-steinberg",
    brightness: float = 1.0,
    contrast: float = 1.0,
    invert: bool = False,
) -> str:
    """Convert an image to ASCII/Unicode art.
    
    Args:
        image_path: Path to image file
        style: Art style (minimal, detailed, artistic)
        character_set: Character set to use for conversion
        width: Target width in characters
        height: Optional target height (preserves aspect if None)
        color_mode: Terminal color depth (1, 8, 256, 16M)
        dithering: Dithering algorithm to use
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor
        invert: Whether to invert the image
        
    Returns:
        Formatted ASCII/Unicode art as a string
        
    Raises:
        ImportError: If PIL is not installed
        FileNotFoundError: If image file doesn't exist
        ValueError: For invalid parameters
    """
    if not HAS_PIL:
        raise ImportError("ASCII art conversion requires Pillow. Install with 'pip install Pillow'")
    
    # Validate parameters
    if character_set not in CHAR_SETS:
        available = ", ".join(CHAR_SETS.keys())
        raise ValueError(f"Invalid character set. Available options: {available}")
    
    if dithering not in DITHERING_ALGORITHMS:
        available = ", ".join(DITHERING_ALGORITHMS.keys())
        raise ValueError(f"Invalid dithering algorithm. Available options: {available}")
    
    # Load and prepare image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Apply style-specific preprocessing
    if style == "minimal":
        # Higher contrast, simplified
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(contrast * 1.5)
    elif style == "detailed":
        # Balanced transformation with enhanced details
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
    elif style == "artistic":
        # Artistic preprocessing with edge enhancement
        img = ImageEnhance.Brightness(img).enhance(brightness * 0.9)
        img = ImageEnhance.Contrast(img).enhance(contrast * 1.2)
        # Additional artistic processing would go here
    else:
        raise ValueError(f"Invalid style: {style}. Choose 'minimal', 'detailed', or 'artistic'")
    
    # Calculate dimensions
    original_width, original_height = img.size
    aspect_ratio = original_height / original_width
    
    if height is None:
        # Preserve aspect ratio by calculating height
        # For ASCII art, each character is typically taller than wide
        # Apply a 0.5 correction factor to account for this
        height = max(1, int(width * aspect_ratio * 0.5))
    
    # Resize image
    img = img.resize((width, height), Image.LANCZOS)
    
    if invert:
        img = ImageOps.invert(img)
    
    # Get characters based on set
    chars = CHAR_SETS[character_set]
    
    # Apply dithering if using monochrome mode
    if color_mode == 1:
        img = DITHERING_ALGORITHMS[dithering].apply(img)
    
    # Generate ASCII art
    lines = []
    pixels = img.load()
    
    for y in range(height):
        line = []
        for x in range(width):
            try:
                if img.mode == "RGBA":
                    r, g, b, a = pixels[x, y]
                    if a < 128:  # Handle transparency
                        line.append(" ")
                        continue
                elif img.mode == "RGB":
                    r, g, b = pixels[x, y]
                elif img.mode == "L":
                    r = g = b = pixels[x, y]
                else:
                    # Convert to RGB for consistency
                    rgb_img = img.convert("RGB")
                    pixels = rgb_img.load()
                    r, g, b = pixels[x, y]
                
                # Map brightness to character
                brightness_value = (r + g + b) / 3  # Average
                char_idx = int((brightness_value / 255) * (len(chars) - 1))
                char = chars[char_idx]
                
                # Apply terminal colors if color mode supports it
                if color_mode > 1:
                    char = apply_terminal_color(char, r, g, b, color_mode)
                
                line.append(char)
            except IndexError:
                line.append(" ")  # Fallback for any indexing issues
        
        lines.append("".join(line))
    
    return "\n".join(lines)

# ╭──────────────────────────────────────────────────────╮
# │  🛠️ Utility Functions - Precision Tools             │
# ╰──────────────────────────────────────────────────────╯

@lru_cache(maxsize=32)
def get_terminal_size() -> Tuple[int, int]:
    """Get terminal dimensions with caching for performance.
    
    Returns:
        Tuple of (width, height) in characters
    """
    try:
        # Standard way to get terminal size
        import shutil
        columns, lines = shutil.get_terminal_size()
        return (columns, lines)
    except (ImportError, AttributeError):
        # Fallback for environments where shutil method isn't available
        try:
            import os
            return (os.get_terminal_size().columns, os.get_terminal_size().lines)
        except (ImportError, AttributeError):
            # Default fallback values
            return (80, 24)

def detect_color_support() -> int:
    """Detect terminal color support level.
    
    Returns:
        Color depth: 1 (monochrome), 8, 16, 256, or 16777216 (16M)
    """
    # Check environment variables for color support
    term = os.environ.get('TERM', '')
    colorterm = os.environ.get('COLORTERM', '')
    
    # True color detection
    if colorterm in ('truecolor', '24bit'):
        return 16777216  # 16M colors
    
    # 256 color detection
    if '256' in term:
        return 256
    
    # Check for basic color terminals
    if term in ('xterm', 'vt100', 'screen', 'ansi'):
        return 8
    
    # Default to monochrome if unsure
    return 1

def create_figlet_art(text: str, font: str = "standard") -> str:
    """Create ASCII text art using Figlet-style fonts.
    
    Args:
        text: Text to convert
        font: Name of the font
        
    Returns:
        ASCII art representation of the text
        
    Note:
        This implementation is designed to work without external dependencies.
    """
    # Basic implementation - would be expanded with more fonts in production
    fonts = {
        "standard": {
            'A': [' *** ', '*   *', '*****', '*   *', '*   *'],
            'B': ['**** ', '*   *', '**** ', '*   *', '**** '],
            'C': [' ****', '*    ', '*    ', '*    ', ' ****'],
            # Additional characters would be defined here
        }
    }
    
    if font not in fonts:
        raise ValueError(f"Font '{font}' not available. Available fonts: {', '.join(fonts.keys())}")
    
    chosen_font = fonts[font]
    text = text.upper()  # Convert to uppercase for simple fonts
    
    # Initialize result
    result = [''] * 5  # Assuming 5 lines per character
    
    for char in text:
        if char in chosen_font:
            for i, line in enumerate(chosen_font[char]):
                result[i] += line
        else:
            # For unsupported characters, add space
            for i in range(5):
                result[i] += '     '
    
    return '\n'.join(result)

def create_banner(title: str, width: int = 40, char: str = '*') -> str:
    """Create a simple banner with a title.
    
    Args:
        title: Banner title text
        width: Banner width
        char: Character used for border
        
    Returns:
        Formatted banner string
    """
    if len(title) + 4 > width:
        title = title[:width-7] + "..."
    
    padding = ' ' * ((width - len(title) - 4) // 2)
    return (
        f"{char * width}\n"
        f"{char} {padding}{title}{padding} {char}\n"
        f"{char * width}"
    )

# ╭──────────────────────────────────────────────────────╮
# │  🔌 Integration Point - Public Interface             │
# ╰──────────────────────────────────────────────────────╯

def transform_image(
    source: Union[str, Path],
    **options
) -> str:
    """High-level function to transform images to terminal art.
    
    A convenience wrapper around the convert function that automatically
    detects optimal parameters based on the terminal environment.
    
    Args:
        source: Path to image file
        **options: Override any convert() parameters
        
    Returns:
        ASCII/Unicode art as a string
    """
    # Auto-detect terminal capabilities if not specified
    terminal_width, _ = get_terminal_size()
    
    # Default parameters
    params = {
        'width': min(terminal_width - 2, 120),  # Leave some margin
        'character_set': 'blocks',  # Unicode blocks for better resolution
        'color_mode': detect_color_support(),
        'style': 'detailed',
    }
    
    # Override with user-provided options
    params.update(options)
    
    # Execute conversion with optimized parameters
    return convert(source, **params)

# ╭──────────────────────────────────────────────────────╮
# │  🧠 Image Processing - Visual Intelligence Core      │
# ╰──────────────────────────────────────────────────────╯

def load_image(source: ImageSource) -> 'Image.Image':
    """Load image from various source types with intelligent handling.
    
    Accepts file paths (string/Path), PIL Image objects, or raw bytes data.
    Performs validation checks and provides clear error messages.
    
    Args:
        source: Image source (file path, PIL Image, or bytes)
        
    Returns:
        PIL Image object ready for processing
        
    Raises:
        ImportError: If PIL is not installed
        ValueError: If source type is invalid or image data corrupt
        FileNotFoundError: If image file doesn't exist
    """
    if not HAS_PIL:
        raise ImportError("Image processing requires Pillow. Install with 'pip install Pillow'")
    
    # Handle different source types
    if isinstance(source, (str, Path)):
        # Load from file path
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {source}")
        
        try:
            return Image.open(path)
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")
            
    elif isinstance(source, bytes):
        # Load from bytes (e.g., from memory or network)
        try:
            import io
            return Image.open(io.BytesIO(source))
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
            
    elif hasattr(source, 'mode') and hasattr(source, 'size'):
        # Already a PIL Image object
        return source
    
    raise ValueError(f"Unsupported image source type: {type(source)}")


def preprocess_image(
    image: 'Image.Image',
    style: str = "detailed",
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    invert: bool = False,
    **options
) -> 'Image.Image':
    """Apply intelligent preprocessing based on style and image characteristics.
    
    Style-aware preprocessing pipeline with automatic adjustment detection.
    
    Args:
        image: PIL Image to process
        style: Art style to apply ("minimal", "detailed", "artistic", "edge", "sketch")
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor
        sharpness: Sharpness adjustment factor
        invert: Whether to invert the image
        **options: Additional options for specific styles
        
    Returns:
        Processed PIL Image
    """
    img = image.copy()  # Work on a copy to preserve original
    
    # Apply style-specific preprocessing
    if style == "minimal":
        # Higher contrast, simplified
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(contrast * 1.5)
        
    elif style == "detailed":
        # Balanced transformation with enhanced details
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
    elif style == "artistic":
        # Artistic preprocessing with edge enhancement
        img = ImageEnhance.Brightness(img).enhance(brightness * 0.9)
        img = ImageEnhance.Contrast(img).enhance(contrast * 1.2)
        img = ImageEnhance.Sharpness(img).enhance(sharpness * 1.5)
        
        # Apply slight color vibrance enhancement
        if img.mode == "RGB" or img.mode == "RGBA":
            from PIL import ImageChops
            
            # Create more vibrant colors by increasing saturation
            # This is done by creating an overlay of the image with itself
            overlay = ImageChops.multiply(img, img)
            factor = options.get('vibrancy', 0.3)
            img = Image.blend(img, overlay, factor)
    
    elif style == "edge":
        # Edge detection for line art effect
        img = img.convert("L")  # Convert to grayscale
        try:
            from PIL import ImageFilter
            # Apply edge detection filter
            edge_img = img.filter(ImageFilter.FIND_EDGES)
            # Enhance the edges
            edge_img = ImageEnhance.Contrast(edge_img).enhance(contrast * 2)
            # Invert for a proper line-art effect (lines are dark on light)
            img = ImageOps.invert(edge_img)
        except (ImportError, AttributeError):
            # Fallback if advanced filters aren't available
            img = ImageEnhance.Contrast(img).enhance(contrast * 1.8)
            
    elif style == "sketch":
        # Pencil sketch effect
        img = img.convert("L")  # Convert to grayscale
        try:
            from PIL import ImageFilter, ImageChops
            
            # Create blurred version
            blur = img.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Invert the blurred image
            blur_inverted = ImageOps.invert(blur)
            
            # Blend with original using color dodge blend mode
            sketch = ImageChops.divide(img, blur_inverted.point(lambda p: p if p > 0 else 1))
            
            img = sketch
        except (ImportError, AttributeError, ZeroDivisionError):
            # Fallback if advanced filters aren't available
            img = ImageEnhance.Contrast(img).enhance(contrast * 1.5)
    
    # Apply universal adjustments
    if invert:
        img = ImageOps.invert(img)
    
    return img


def resize_image_with_aspect(
    image: 'Image.Image',
    width: int,
    height: Optional[int] = None,
    char_aspect_ratio: float = 0.5
) -> 'Image.Image':
    """Resize image with proper aspect ratio preservation for ASCII art.
    
    Terminal characters are typically taller than wide, so this function
    applies a correction factor to maintain visual aspect ratio.
    
    Args:
        image: PIL Image to resize
        width: Target width in characters
        height: Optional target height (preserves aspect if None)
        char_aspect_ratio: Width-to-height ratio of a character in terminal
        
    Returns:
        Resized PIL Image
    """
    original_width, original_height = image.size
    aspect_ratio = original_height / original_width
    
    # If height is not specified, calculate it based on width
    # Character cells in terminals are typically taller than wide
    # so we apply the char_aspect_ratio correction factor
    if height is None:
        height = max(1, int(width * aspect_ratio / char_aspect_ratio))
    
    # Ensure minimum dimensions
    width = max(1, width)
    height = max(1, height)
    
    # Use high quality resampling methods
    try:
        return image.resize((width, height), Image.LANCZOS)
    except (AttributeError, ValueError):
        # Fallback for older PIL versions
        return image.resize((width, height), Image.BICUBIC)


class AsciiRenderer:
    """High-performance ASCII art renderer with advanced character mapping.
    
    This class handles the core conversion from pixel data to characters with
    optimized algorithms and customizable rendering strategies.
    """
    
    def __init__(self, 
                 character_set: str = "standard", 
                 custom_chars: Optional[str] = None,
                 brightness_strategy: str = "average",
                 invert_mapping: bool = False):
        """Initialize the ASCII renderer with specific parameters.
        
        Args:
            character_set: Name of predefined character set or "custom"
            custom_chars: Custom character sequence (used if character_set is "custom")
            brightness_strategy: How pixel brightness is calculated ("average", "luminance", "max")
            invert_mapping: Whether to invert the brightness-to-character mapping
        """
        # Set up character mapping
        if character_set == "custom" and custom_chars:
            self.chars = custom_chars
        elif character_set in ALL_CHAR_SETS:
            self.chars = ALL_CHAR_SETS[character_set]
        else:
            available = ", ".join(ALL_CHAR_SETS.keys())
            raise ValueError(f"Invalid character set. Available options: {available}")
        
        # Invert character mapping if requested
        if invert_mapping:
            self.chars = self.chars[::-1]
        
        # Set brightness calculation strategy
        brightness_strategies = {
            "average": lambda r, g, b: (r + g + b) / 3,
            "luminance": lambda r, g, b: 0.299 * r + 0.587 * g + 0.114 * b,
            "max": lambda r, g, b: max(r, g, b)
        }
        
        if brightness_strategy not in brightness_strategies:
            raise ValueError("Brightness strategy must be 'average', 'luminance', or 'max'")
        
        self.calc_brightness = brightness_strategies[brightness_strategy]
    
    def map_pixel_to_char(self, pixel: Union[Tuple[int, ...], int]) -> str:
        """Map a pixel to a character based on brightness.
        
        Args:
            pixel: Pixel value (RGB tuple or grayscale integer)
            
        Returns:
            Corresponding character from the character set
        """
        # Extract brightness based on pixel type
        if isinstance(pixel, (tuple, list)):
            if len(pixel) >= 3:  # RGB or RGBA
                r, g, b = pixel[:3]
                brightness = self.calc_brightness(r, g, b)
            else:
                brightness = pixel[0]  # Grayscale in tuple form
        else:
            brightness = pixel  # Direct grayscale value
        
        # Map brightness to character index
        char_idx = int((brightness / 255) * (len(self.chars) - 1))
        # Ensure index is in bounds
        char_idx = max(0, min(char_idx, len(self.chars) - 1))
        
        return self.chars[char_idx]
    
    def render_image(self, 
                     image: 'Image.Image', 
                     color_mode: int = 8,
                     bg_color: Optional[Tuple[int, int, int]] = None) -> List[str]:
        """Render an image to ASCII art with optional color.
        
        Args:
            image: PIL Image to render
            color_mode: Terminal color depth
            bg_color: Optional background color
            
        Returns:
            List of strings representing ASCII art rows
        """
        width, height = image.size
        pixels = image.load()
        
        # Prepare output
        lines = []
        
        # Process each row
        for y in range(height):
            line = []
            # Process each column in the row
            for x in range(width):
                try:
                    # Extract pixel data based on image mode
                    if image.mode == "RGBA":
                        r, g, b, a = pixels[x, y]
                        if a < 128:  # Handle transparency
                            line.append(" ")
                            continue
                    elif image.mode == "RGB":
                        r, g, b = pixels[x, y]
                    elif image.mode == "L":
                        r = g = b = pixels[x, y]
                    else:
                        # For other modes, ensure we have RGB data
                        rgb_val = image.getpixel((x, y))
                        if isinstance(rgb_val, (tuple, list)) and len(rgb_val) >= 3:
                            r, g, b = rgb_val[:3]
                        else:
                            r = g = b = rgb_val
                    
                    # Map pixel to character
                    char = self.map_pixel_to_char((r, g, b))
                    
                    # Apply terminal colors if in color mode
                    if color_mode > 1:
                        char = apply_terminal_color(char, r, g, b, color_mode, bg_color)
                    
                    line.append(char)
                except (IndexError, TypeError):
                    # Fallback for any error
                    line.append(" ")
            
            # Add completed line
            lines.append("".join(line))
        
        return lines


# Enhanced core transformation function
def convert(
    image_source: ImageSource,
    style: str = "detailed",
    character_set: str = "standard",
    width: int = 80,
    height: Optional[int] = None,
    color_mode: int = 8,
    dithering: str = "none",
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    invert: bool = False,
    bg_color: Optional[Tuple[int, int, int]] = None,
    brightness_strategy: str = "luminance",
    char_aspect_ratio: float = 0.5,
    custom_chars: Optional[str] = None,
    **extra_options
) -> str:
    """Convert an image to ASCII/Unicode art with intelligent optimization.
    
    Enhanced transformation engine with comprehensive preprocessing, rendering
    control, and automatic optimization for different image types.
    
    Args:
        image_source: Source image (path, PIL Image, or bytes)
        style: Art style ("minimal", "detailed", "artistic", "edge", "sketch")
        character_set: Character set name or "custom"
        width: Target width in characters
        height: Optional target height (preserves aspect if None)
        color_mode: Terminal color depth (1, 8, 16, 256, 16M)
        dithering: Dithering algorithm for monochrome mode
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor
        sharpness: Sharpness adjustment factor
        invert: Whether to invert the image
        bg_color: Optional background color for characters
        brightness_strategy: Strategy for calculating pixel brightness
        char_aspect_ratio: Width-to-height ratio of a character
        custom_chars: Custom character sequence (for "custom" character_set)
        **extra_options: Additional options passed to preprocessing
        
    Returns:
        Formatted ASCII/Unicode art as a string
        
    Raises:
        ImportError: If PIL is not installed
        ValueError: For invalid parameters
        FileNotFoundError: If image file doesn't exist
    """
    # Load image from source
    img = load_image(image_source)
    
    # Apply preprocessing based on style
    img = preprocess_image(
        img, style, brightness, contrast, sharpness, invert, **extra_options
    )
    
    # Resize with aspect ratio correction
    img = resize_image_with_aspect(img, width, height, char_aspect_ratio)
    
    # Apply dithering in monochrome mode
    if color_mode == 1 and dithering != "none":
        if dithering not in DITHERING_ALGORITHMS:
            available = ", ".join(DITHERING_ALGORITHMS.keys())
            raise ValueError(f"Invalid dithering algorithm. Available: {available}")
        img = DITHERING_ALGORITHMS[dithering].apply(img)
    
    # Create renderer with character set
    renderer = AsciiRenderer(
        character_set=character_set,
        custom_chars=custom_chars,
        brightness_strategy=brightness_strategy,
        invert_mapping=False  # Already handled by preprocess_image
    )
    
    # Render image to ASCII lines
    lines = renderer.render_image(img, color_mode, bg_color)
    
    # Join into final output
    return "\n".join(lines)


# ╭──────────────────────────────────────────────────────╮
# │  🔌 Integration Point - Public Interface             │
# ╰──────────────────────────────────────────────────────╯

def transform_image(
    source: ImageSource,
    **options
) -> str:
    """High-level function to transform images to terminal art with intelligent defaults.
    
    Automatically detects optimal parameters based on terminal environment and image content.
    
    Args:
        source: Image source (path, PIL Image object, or bytes)
        **options: Override any convert() parameters
        
    Returns:
        ASCII/Unicode art as a string
    """
    # Auto-detect terminal capabilities
    terminal_width, terminal_height = get_terminal_size()
    color_depth = detect_color_support()
    
    # Load image to analyze content (if not already a PIL Image)
    img = None
    if not isinstance(source, (str, Path, bytes)):
        # Already a PIL Image
        img = source
    
    # Determine optimal parameters based on image and environment
    params = {
        # Size parameters - leave margin on terminal width
        'width': min(terminal_width - 2, 120),
        
        # Visual parameters - select based on terminal capabilities
        'character_set': 'blocks' if color_depth >= 8 else 'standard',
        'color_mode': color_depth,
        'style': 'detailed',
        
        # Use dithering in monochrome mode
        'dithering': 'floyd-steinberg' if color_depth == 1 else 'none',
    }
    
    # Analyze image content if available to make better choices
    if img and HAS_PIL:
        # For photos with lots of detail, use more detailed character sets
        try:
            # Measure image complexity (edge density)
            from PIL import ImageFilter
            edge_img = img.convert("L").filter(ImageFilter.FIND_EDGES)
            edge_density = sum(edge_img.getdata()) / (255 * img.width * img.height)
            
            # For detailed images, use more refined character set
            if edge_density > 0.1:  # Moderate detail
                if color_depth <= 1:
                    params['character_set'] = 'extended'
                    params['brightness_strategy'] = 'luminance'
            
            # For very simple images, use minimal character set
            elif edge_density < 0.05:  # Low detail
                params['character_set'] = 'minimal' if color_depth <= 1 else 'blocks'
            
        except (ImportError, AttributeError, TypeError):
            pass  # Fallback to defaults if analysis fails
    
    # Override with user-provided options
    params.update(options)
    
    # Execute conversion with optimized parameters
    return convert(source, **params)


class AsciiArtBuilder:
    """Fluent interface for building and configuring ASCII art transformations.
    
    Provides a chainable API for step-by-step configuration of ASCII art generation
    with intelligent defaults and composition patterns.
    
    Example:
        art = AsciiArtBuilder("image.jpg")
                .width(60)
                .style("artistic")
                .character_set("blocks")
                .color()
                .render()
    """
    
    def __init__(self, source: Optional[ImageSource] = None):
        """Initialize the builder with optional image source.
        
        Args:
            source: Image source (path, PIL Image, or bytes)
        """
        self.source = source
        self.options = {}
        self.result = None
    
    def from_source(self, source: ImageSource) -> 'AsciiArtBuilder':
        """Set the image source.
        
        Args:
            source: Image source (path, PIL Image, or bytes)
            
        Returns:
            Self for chaining
        """
        self.source = source
        return self
    
    def width(self, width: int) -> 'AsciiArtBuilder':
        """Set the output width in characters.
        
        Args:
            width: Width in characters
            
        Returns:
            Self for chaining
        """
        self.options['width'] = width
        return self
    
    def height(self, height: int) -> 'AsciiArtBuilder':
        """Set the output height in characters.
        
        Args:
            height: Height in characters
            
        Returns:
            Self for chaining
        """
        self.options['height'] = height
        return self
    
    def style(self, style: str) -> 'AsciiArtBuilder':
        """Set the art style.
        
        Args:
            style: Art style ("minimal", "detailed", "artistic", "edge", "sketch")
            
        Returns:
            Self for chaining
        """
        self.options['style'] = style
        return self
    
    def character_set(self, char_set: str) -> 'AsciiArtBuilder':
        """Set the character set.
        
        Args:
            char_set: Character set name or "custom"
            
        Returns:
            Self for chaining
        """
        self.options['character_set'] = char_set
        return self
    
    def custom_chars(self, chars: str) -> 'AsciiArtBuilder':
        """Set custom character sequence.
        
        Args:
            chars: Custom character sequence (darkest to lightest)
            
        Returns:
            Self for chaining
        """
        self.options['character_set'] = 'custom'
        self.options['custom_chars'] = chars
        return self
    
    def color(self, mode: int = -1) -> 'AsciiArtBuilder':
        """Enable color output with optional mode specification.
        
        Args:
            mode: Color mode (-1 for auto-detect, 1 for mono, 8, 16, 256, 16M)
            
        Returns:
            Self for chaining
        """
        if mode < 0:
            # Auto-detect
            self.options['color_mode'] = detect_color_support()
        else:
            self.options['color_mode'] = mode
        return self
    
    def monochrome(self, dither: str = "floyd-steinberg") -> 'AsciiArtBuilder':
        """Set monochrome mode with dithering.
        
        Args:
            dither: Dithering algorithm
            
        Returns:
            Self for chaining
        """
        self.options['color_mode'] = 1
        self.options['dithering'] = dither
        return self
    
    def brightness(self, value: float) -> 'AsciiArtBuilder':
        """Set brightness adjustment.
        
        Args:
            value: Brightness factor (0.0-2.0, 1.0 is normal)
            
        Returns:
            Self for chaining
        """
        self.options['brightness'] = value
        return self
    
    def contrast(self, value: float) -> 'AsciiArtBuilder':
        """Set contrast adjustment.
        
        Args:
            value: Contrast factor (0.0-2.0, 1.0 is normal)
            
        Returns:
            Self for chaining
        """
        self.options['contrast'] = value
        return self
    
    def invert(self, value: bool = True) -> 'AsciiArtBuilder':
        """Set image inversion.
        
        Args:
            value: Whether to invert the image
            
        Returns:
            Self for chaining
        """
        self.options['invert'] = value
        return self
    
    def render(self) -> str:
        """Render the ASCII art with current configuration.
        
        Returns:
            ASCII art string
            
        Raises:
            ValueError: If no source has been specified
        """
        if self.source is None:
            raise ValueError("No image source specified")
        
        self.result = transform_image(self.source, **self.options)
        return self.result
    
    def save(self, path: Union[str, Path]) -> 'AsciiArtBuilder':
        """Save the rendered ASCII art to a file.
        
        Renders the art if not already rendered.
        
        Args:
            path: Output file path
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If no source has been specified
        """
        if self.result is None:
            self.render()
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.result)
        
        return self
    
    def print(self) -> 'AsciiArtBuilder':
        """Print the rendered ASCII art to console.
        
        Renders the art if not already rendered.
        
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If no source has been specified
        """
        if self.result is None:
            self.render()
        
        print(self.result)
        return self

# Create shorthand function for fluent API
def art(source: Optional[ImageSource] = None) -> AsciiArtBuilder:
    """Create an AsciiArtBuilder for fluent configuration.
    
    Args:
        source: Optional image source
        
    Returns:
        AsciiArtBuilder instance
    """
    return AsciiArtBuilder(source)

