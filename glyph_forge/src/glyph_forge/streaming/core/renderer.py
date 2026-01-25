"""
High-Fidelity Glyph Renderer for Glyph Forge.

Converts video frames to beautiful terminal-displayable glyph art using:
- Gradient rendering (block characters)
- Vectorized operations (numpy)
- Cached lookup tables
- Multiple color modes (ANSI256, TrueColor)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache
import pickle
import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════
# Character Sets
# ═══════════════════════════════════════════════════════════════

# Extended gradient - sorted by visual density
# Dense characters for dark pixels → sparse for bright (works on dark terminals)
GRADIENT_CHARS = '█▓▒░·˙ '

# Extended gradient with 500+ levels for maximum fidelity
EXTENDED_GRADIENT = (
    '█' * 50 + '▓' * 50 +
    '▓' * 25 + '▒' * 25 +
    '▒' * 50 +
    '▒' * 25 + '░' * 25 +
    '░' * 50 +
    '░' * 25 + '·' * 25 +
    '·' * 25 + '˙' * 25 +
    '˙' * 25 + ' ' * 25 +
    ' ' * 58
)

# ASCII-only fallback
ASCII_GRADIENT = '@%#*+=-:. '

# Braille patterns base (U+2800)
BRAILLE_BASE = 0x2800


# ═══════════════════════════════════════════════════════════════
# Lookup Tables
# ═══════════════════════════════════════════════════════════════

class LookupTables:
    """Pre-computed lookup tables for fast rendering.
    
    Tables are cached to disk for instant startup after first run.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize lookup tables.
        
        Args:
            cache_dir: Directory for cached tables
        """
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'glyph_forge'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / 'lookup_tables_v2.pkl'
        
        self._luminance_to_char: Optional[np.ndarray] = None
        self._rgb_to_ansi256: Optional[np.ndarray] = None
        
        self._load_or_build()
    
    def _load_or_build(self):
        """Load tables from cache or build them."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self._luminance_to_char = data['luminance_to_char']
                    self._rgb_to_ansi256 = data.get('rgb_to_ansi256')
                    print(f"✓ Loaded lookup tables from cache")
                    return
            except Exception as e:
                print(f"Cache load failed: {e}")
        
        print("Building lookup tables (one-time)...")
        self._build_tables()
        self._save_cache()
    
    def _build_tables(self):
        """Build all lookup tables."""
        # Luminance → character (256 levels → char index)
        self._luminance_to_char = np.zeros(256, dtype=np.uint8)
        
        for lum in range(256):
            # Map luminance to gradient position
            # Use extended gradient for higher fidelity
            idx = int(lum / 256 * len(EXTENDED_GRADIENT))
            idx = min(idx, len(EXTENDED_GRADIENT) - 1)
            self._luminance_to_char[lum] = idx
        
        # RGB → ANSI256 (for fast color mapping)
        # Build 3D lookup table: 16x16x16 (reduced from 256^3 for memory)
        self._rgb_to_ansi256 = np.zeros((16, 16, 16), dtype=np.uint8)
        
        for r in range(16):
            for g in range(16):
                for b in range(16):
                    # Scale up to 0-255
                    R, G, B = r * 17, g * 17, b * 17
                    self._rgb_to_ansi256[r, g, b] = self._compute_ansi256(R, G, B)
    
    def _compute_ansi256(self, r: int, g: int, b: int) -> int:
        """Compute closest ANSI256 color code."""
        # Check grayscale ramp (232-255)
        gray = (r + g + b) // 3
        if abs(r - gray) < 10 and abs(g - gray) < 10 and abs(b - gray) < 10:
            # Use grayscale
            gray_idx = int(gray / 255 * 23)
            return 232 + gray_idx
        
        # Use 6x6x6 color cube (16-231)
        r6 = int(r / 256 * 6)
        g6 = int(g / 256 * 6)
        b6 = int(b / 256 * 6)
        return 16 + 36 * r6 + 6 * g6 + b6
    
    def _save_cache(self):
        """Save tables to cache."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'luminance_to_char': self._luminance_to_char,
                    'rgb_to_ansi256': self._rgb_to_ansi256,
                }, f)
            print(f"✓ Saved lookup tables to cache")
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    def get_char_index(self, luminance: np.ndarray) -> np.ndarray:
        """Map luminance values to character indices."""
        return self._luminance_to_char[luminance]
    
    def get_ansi256(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Map RGB values to ANSI256 codes."""
        # Reduce to 4-bit per channel for lookup
        r4 = (r >> 4).astype(np.uint8)
        g4 = (g >> 4).astype(np.uint8)
        b4 = (b >> 4).astype(np.uint8)
        return self._rgb_to_ansi256[r4, g4, b4]


# ═══════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════

@dataclass
class RenderConfig:
    """Renderer configuration."""
    mode: str = 'gradient'  # gradient, braille, ascii, blocks
    color: str = 'ansi256'  # none, ansi256, truecolor
    dithering: bool = False
    cache_dir: Optional[Path] = None


class GlyphRenderer:
    """High-performance glyph renderer.
    
    Converts video frames to terminal-displayable glyph strings.
    Uses vectorized numpy operations and cached lookup tables.
    
    Usage:
        renderer = GlyphRenderer()
        glyph_string = renderer.render(frame, width=120, height=40)
        print(glyph_string)
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize renderer.
        
        Args:
            config: Render configuration
        """
        self.config = config or RenderConfig()
        self.tables = LookupTables(self.config.cache_dir)
        
        # Pre-compute character array for fast lookup
        self._char_array = np.array(list(EXTENDED_GRADIENT), dtype='U1')
        
        # Color code cache
        self._color_cache: Dict[int, str] = {}
    
    def render(
        self,
        frame: np.ndarray,
        width: int = 120,
        height: int = 40,
        color: Optional[str] = None,
    ) -> str:
        """Render frame to glyph string.
        
        Args:
            frame: BGR image (numpy array)
            width: Output width in characters
            height: Output height in characters
            color: Color mode override ('none', 'ansi256', 'truecolor')
            
        Returns:
            Terminal-ready string with ANSI escape codes
        """
        color = color or self.config.color
        
        # Resize frame to target dimensions
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for character mapping
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Get character indices via lookup table
        char_indices = self.tables.get_char_index(gray)
        
        # Build output based on color mode
        if color == 'none':
            return self._render_no_color(char_indices)
        elif color == 'ansi256':
            return self._render_ansi256(frame, char_indices)
        elif color == 'truecolor':
            return self._render_truecolor(frame, char_indices)
        else:
            return self._render_no_color(char_indices)
    
    def _render_no_color(self, char_indices: np.ndarray) -> str:
        """Render without color (fastest)."""
        chars = self._char_array[char_indices]
        
        lines = []
        for row in chars:
            lines.append(''.join(row))
        
        return '\n'.join(lines)
    
    def _render_ansi256(self, frame: np.ndarray, char_indices: np.ndarray) -> str:
        """Render with ANSI256 colors (good balance)."""
        # Get color codes
        if len(frame.shape) == 3:
            b, g, r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        else:
            r = g = b = frame
        
        color_codes = self.tables.get_ansi256(r, g, b)
        chars = self._char_array[char_indices]
        
        # Build output with run-length encoding
        lines = []
        
        for y in range(frame.shape[0]):
            line_parts = []
            last_code = -1  # Reset for each line
            
            for x in range(frame.shape[1]):
                code = int(color_codes[y, x])
                char = chars[y, x]
                
                if code != last_code:
                    line_parts.append(f'\033[38;5;{code}m{char}')
                    last_code = code
                else:
                    line_parts.append(char)
            
            lines.append(''.join(line_parts))
        
        return '\033[0m' + '\n'.join(lines) + '\033[0m'
    
    def _render_truecolor(self, frame: np.ndarray, char_indices: np.ndarray) -> str:
        """Render with 24-bit TrueColor (highest quality)."""
        if len(frame.shape) == 3:
            b, g, r = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        else:
            r = g = b = frame
        
        chars = self._char_array[char_indices]
        
        lines = []
        last_rgb = (-1, -1, -1)
        
        for y in range(frame.shape[0]):
            line_parts = []
            
            for x in range(frame.shape[1]):
                R, G, B = int(r[y, x]), int(g[y, x]), int(b[y, x])
                char = chars[y, x]
                
                if (R, G, B) != last_rgb:
                    line_parts.append(f'\033[38;2;{R};{G};{B}m{char}')
                    last_rgb = (R, G, B)
                else:
                    line_parts.append(char)
            
            lines.append(''.join(line_parts))
        
        return '\033[0m' + '\n'.join(lines) + '\033[0m'
    
    def render_braille(
        self,
        frame: np.ndarray,
        width: int = 120,
        height: int = 40,
        threshold: int = 128,
    ) -> str:
        """Render using Braille characters (2x4 sub-pixel resolution).
        
        Each Braille character represents a 2x4 pixel block, giving
        effectively 2x vertical and 2x horizontal resolution.
        
        Args:
            frame: BGR image
            width: Output width in characters
            height: Output height in characters
            threshold: Binary threshold for dots
            
        Returns:
            Braille-rendered string
        """
        # Braille uses 2x4 blocks, so we need 2x width and 4x height pixels
        pixel_w = width * 2
        pixel_h = height * 4
        
        # Resize and convert to grayscale
        frame = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_AREA)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Binary threshold
        binary = (gray > threshold).astype(np.uint8)
        
        # Braille dot positions (column-major):
        # 0 3
        # 1 4
        # 2 5
        # 6 7
        dot_weights = np.array([
            [0x01, 0x08],  # Row 0: dots 0, 3
            [0x02, 0x10],  # Row 1: dots 1, 4
            [0x04, 0x20],  # Row 2: dots 2, 5
            [0x40, 0x80],  # Row 3: dots 6, 7
        ], dtype=np.uint8)
        
        lines = []
        for char_y in range(height):
            row_chars = []
            for char_x in range(width):
                # Extract 2x4 block
                py = char_y * 4
                px = char_x * 2
                
                block = binary[py:py+4, px:px+2]
                
                # Calculate Braille code
                code = 0
                for dy in range(min(4, block.shape[0])):
                    for dx in range(min(2, block.shape[1])):
                        if block[dy, dx]:
                            code |= dot_weights[dy, dx]
                
                row_chars.append(chr(BRAILLE_BASE + code))
            
            lines.append(''.join(row_chars))
        
        return '\n'.join(lines)
