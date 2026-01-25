"""Character rendering and glyph mapping.

This module provides character set definitions and rendering functions
for converting processed frame data into Unicode/ASCII art.

Classes:
    CharacterRenderer: Renders processed frames to character art
    FrameRenderer: Renders complete frames with borders and stats
    FrameBuffer: Thread-safe frame buffer for display

Constants:
    CHARACTER_MAPS: Default character maps for different styles
"""
from __future__ import annotations

import collections
import math
import os
import re
import threading
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import QualityLevel, StreamMetrics, RenderParameters


# ═══════════════════════════════════════════════════════════════
# Character Maps
# ═══════════════════════════════════════════════════════════════

CHARACTER_MAPS: Dict[str, Dict[str, Any]] = {
    # Density gradients (dark to light)
    "gradients": {
        "standard": "█▓▒░ ",
        "enhanced": "█▇▆▅▄▃▂▁▀ ",
        "blocks": "█▉▊▋▌▍▎▏ ",
        "braille": "⣿⣷⣯⣟⡿⢿⣻⣽⣾ ",
        "ascii": "@%#*+=-:. ",
        "detailed": "█▓▒░■◆●◉○♦♥♠☻☺⬢⬡✿❀✣❄★☆✩✫✬ .  ",
        "minimal": "█▓░ ",
        "dots": "⣿⢿⡿⣟⣯⣷⣾⣽⣻ ",
        "shades": "██▓▓▒▒░░  ",
    },
    
    # Edge characters by direction
    "edges": {
        "horizontal": {"bold": "━", "standard": "─", "light": "╌", "ascii": "-"},
        "vertical": {"bold": "┃", "standard": "│", "light": "╎", "ascii": "|"},
        "diagonal_ne": {"bold": "╱", "standard": "╱", "light": "╱", "ascii": "/"},
        "diagonal_nw": {"bold": "╲", "standard": "╲", "light": "╲", "ascii": "\\"},
    },
    
    # Border characters
    "borders": {
        "unicode": {
            "top_left": "╔",
            "top_right": "╗",
            "bottom_left": "╚",
            "bottom_right": "╝",
            "horizontal": "═",
            "vertical": "║",
        },
        "ascii": {
            "top_left": "+",
            "top_right": "+",
            "bottom_left": "+",
            "bottom_right": "+",
            "horizontal": "-",
            "vertical": "|",
        },
    },
}


# ═══════════════════════════════════════════════════════════════
# Character Renderer
# ═══════════════════════════════════════════════════════════════

class CharacterRenderer:
    """Renders processed frame data to character art.
    
    Converts block luminance, color, and edge data into Unicode or
    ASCII character art with optional ANSI color support.
    
    Attributes:
        gradient: Character gradient string (dense to sparse)
        use_unicode: Whether to use Unicode characters
        use_color: Whether to output ANSI colors
        edge_mode: Edge rendering mode (none, basic, enhanced)
    """
    
    def __init__(
        self,
        gradient: str = "standard",
        use_unicode: bool = True,
        use_color: bool = True,
        edge_mode: str = "enhanced",
        edge_threshold: int = 30,
    ):
        """Initialize character renderer.
        
        Args:
            gradient: Gradient preset name or custom gradient string
            use_unicode: Enable Unicode characters
            use_color: Enable ANSI color output
            edge_mode: Edge rendering (none, basic, enhanced)
            edge_threshold: Edge detection threshold for character selection
        """
        self.use_unicode = use_unicode
        self.use_color = use_color
        self.edge_mode = edge_mode
        self.edge_threshold = edge_threshold
        
        # Resolve gradient
        if gradient in CHARACTER_MAPS["gradients"]:
            self._gradient = CHARACTER_MAPS["gradients"][gradient]
        else:
            self._gradient = gradient if gradient else CHARACTER_MAPS["gradients"]["standard"]
        
        # Select edge chars based on unicode support
        edge_style = "standard" if use_unicode else "ascii"
        self._edge_chars = {
            k: v[edge_style] for k, v in CHARACTER_MAPS["edges"].items()
        }
        
        # Color cache
        self._color_cache: Dict[Tuple[int, int, int], str] = {}
        self._lock = threading.RLock()
    
    def render(
        self,
        processed_data: Dict[str, Any],
        show_edges: bool = True,
    ) -> List[str]:
        """Render processed frame data to character art.
        
        Args:
            processed_data: Output from FrameProcessor.process_frame()
            show_edges: Whether to show edge characters
            
        Returns:
            List of strings, one per line
        """
        blocks = processed_data.get("blocks")
        colors = processed_data.get("colors")
        edges = processed_data.get("edges")
        height = processed_data.get("height", 0)
        width = processed_data.get("width", 0)
        
        if blocks is None or height == 0 or width == 0:
            return []
        
        lines = []
        
        for y in range(height):
            line_chars = []
            
            for x in range(width):
                density = blocks[y, x]
                
                # Check for edge character
                char = None
                if show_edges and edges is not None and self.edge_mode != "none":
                    edge_mag = edges[y, x, 0]
                    if edge_mag > self.edge_threshold:
                        grad_x = edges[y, x, 1]
                        grad_y = edges[y, x, 2]
                        char = self._get_edge_char(grad_x, grad_y, edge_mag)
                
                # Fall back to density character
                if char is None:
                    char = self._get_density_char(density)
                
                # Apply color if available
                if self.use_color and colors is not None:
                    r, g, b = colors[y, x]
                    char = self._apply_color(char, r, g, b)
                
                line_chars.append(char)
            
            # Join and add reset if using color
            line = "".join(line_chars)
            if self.use_color:
                line += "\033[0m"
            
            lines.append(line)
        
        return lines
    
    def _get_density_char(self, density: float) -> str:
        """Get character for luminance density value.
        
        Args:
            density: Normalized density (0.0=dark, 1.0=bright)
            
        Returns:
            Single character
        """
        # Invert density (high luminance = sparse character)
        idx = int((1.0 - density) * (len(self._gradient) - 1))
        idx = max(0, min(len(self._gradient) - 1, idx))
        return self._gradient[idx]
    
    def _get_edge_char(
        self,
        grad_x: float,
        grad_y: float,
        magnitude: float,
    ) -> Optional[str]:
        """Get edge character based on gradient direction.
        
        Args:
            grad_x: X gradient component
            grad_y: Y gradient component
            magnitude: Edge magnitude
            
        Returns:
            Edge character or None
        """
        if abs(grad_x) < 1e-6 and abs(grad_y) < 1e-6:
            return None
        
        # Calculate angle
        angle = math.degrees(math.atan2(grad_y, grad_x)) % 180
        
        # Select edge character based on angle
        if angle < 22.5 or angle >= 157.5:
            return self._edge_chars["horizontal"]
        elif 67.5 <= angle < 112.5:
            return self._edge_chars["vertical"]
        elif 22.5 <= angle < 67.5:
            return self._edge_chars["diagonal_ne"]
        else:
            return self._edge_chars["diagonal_nw"]
    
    @lru_cache(maxsize=4096)
    def _get_color_code(self, r: int, g: int, b: int) -> str:
        """Get ANSI 24-bit color code."""
        return f"\033[38;2;{r};{g};{b}m"
    
    def _apply_color(self, char: str, r: int, g: int, b: int) -> str:
        """Apply ANSI color to character.
        
        Args:
            char: Character to colorize
            r, g, b: RGB color values
            
        Returns:
            Colored character string
        """
        color_code = self._get_color_code(r, g, b)
        return f"{color_code}{char}"
    
    def set_gradient(self, gradient: str) -> None:
        """Set the character gradient.
        
        Args:
            gradient: Preset name or custom gradient string
        """
        if gradient in CHARACTER_MAPS["gradients"]:
            self._gradient = CHARACTER_MAPS["gradients"][gradient]
        else:
            self._gradient = gradient


# ═══════════════════════════════════════════════════════════════
# Frame Renderer (Display with borders/stats)
# ═══════════════════════════════════════════════════════════════

class FrameRenderer:
    """Renders complete frames with borders and statistics.
    
    Wraps character art with decorative borders, title, and
    performance statistics display.
    """
    
    def __init__(
        self,
        terminal_width: int = 120,
        terminal_height: int = 40,
        use_unicode: bool = True,
        show_border: bool = True,
        show_stats: bool = True,
    ):
        """Initialize frame renderer.
        
        Args:
            terminal_width: Available terminal width
            terminal_height: Available terminal height
            use_unicode: Use Unicode border characters
            show_border: Display decorative border
            show_stats: Display performance statistics
        """
        self.terminal_width = max(40, terminal_width)
        self.terminal_height = max(10, terminal_height)
        self.show_border = show_border
        self.show_stats = show_stats
        
        # Select border characters
        border_type = "unicode" if use_unicode else "ascii"
        self._borders = CHARACTER_MAPS["borders"][border_type]
        
        # Caches
        self._title_cache: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def render_frame(
        self,
        art_lines: List[str],
        title: str = "",
        metrics: Optional[StreamMetrics] = None,
        params: Optional[RenderParameters] = None,
    ) -> List[str]:
        """Render complete frame with borders and stats.
        
        Args:
            art_lines: Character art lines
            title: Title to display
            metrics: Performance metrics
            params: Render parameters
            
        Returns:
            Complete frame lines
        """
        if not art_lines:
            return []
        
        # Calculate content width (strip ANSI codes for measurement)
        content_width = self._measure_line(art_lines[0]) if art_lines else 0
        
        frame = []
        
        # Add title bar
        if self.show_border and title:
            frame.append(self._format_title(title, content_width))
        
        # Add content
        if self.show_border:
            for line in art_lines:
                frame.append(f"{self._borders['vertical']}{line}{self._borders['vertical']}")
        else:
            frame.extend(art_lines)
        
        # Add stats
        if self.show_stats and metrics:
            if self.show_border:
                # Bottom border
                frame.append(
                    f"{self._borders['bottom_left']}"
                    f"{self._borders['horizontal'] * content_width}"
                    f"{self._borders['bottom_right']}"
                )
            
            stats_line = self._format_stats(metrics, params, content_width)
            if self.show_border:
                frame.append(
                    f"{self._borders['vertical']} {stats_line} {self._borders['vertical']}"
                )
                frame.append(
                    f"{self._borders['bottom_left']}"
                    f"{self._borders['horizontal'] * content_width}"
                    f"{self._borders['bottom_right']}"
                )
            else:
                frame.append(stats_line)
        elif self.show_border:
            # Just bottom border
            frame.append(
                f"{self._borders['bottom_left']}"
                f"{self._borders['horizontal'] * content_width}"
                f"{self._borders['bottom_right']}"
            )
        
        return frame
    
    def _format_title(self, title: str, width: int) -> str:
        """Format title bar."""
        # Truncate title if needed
        if len(title) > width - 4:
            title = title[:width - 7] + "..."
        
        padded_title = f" {title} "
        padding = max(0, width - len(padded_title))
        left_pad = padding // 2
        right_pad = padding - left_pad
        
        return (
            f"{self._borders['top_left']}"
            f"{self._borders['horizontal'] * left_pad}"
            f"{padded_title}"
            f"{self._borders['horizontal'] * right_pad}"
            f"{self._borders['top_right']}"
        )
    
    def _format_stats(
        self,
        metrics: StreamMetrics,
        params: Optional[RenderParameters],
        width: int,
    ) -> str:
        """Format performance statistics."""
        fps = f"FPS: {metrics.current_fps:.1f}"
        render = f"Render: {metrics.average_render_time:.1f}ms"
        
        if params:
            quality = f"Q: {params.quality_level}/4"
            stats = f"{fps} | {render} | {quality}"
        else:
            stats = f"{fps} | {render}"
        
        # Pad to width
        if len(stats) > width - 2:
            stats = stats[:width - 5] + "..."
        
        return stats.ljust(width - 2)
    
    def _measure_line(self, line: str) -> int:
        """Measure visual width of line (excluding ANSI codes)."""
        # Strip ANSI escape sequences
        clean = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
        
        width = 0
        for c in clean:
            if ord(c) < 127:
                width += 1
            elif unicodedata.east_asian_width(c) in ('F', 'W'):
                width += 2
            elif unicodedata.category(c) not in ('Mn', 'Me', 'Cf'):
                width += 1
        
        return width


# ═══════════════════════════════════════════════════════════════
# Frame Buffer
# ═══════════════════════════════════════════════════════════════

class FrameBuffer:
    """Thread-safe frame buffer with timing information.
    
    Stores rendered frames with timestamps for synchronized playback
    at the correct frame rate.
    
    Attributes:
        capacity: Maximum frames to store
        target_fps: Target playback framerate
    """
    
    def __init__(
        self,
        capacity: int = 60,
        target_fps: float = 30.0,
    ):
        """Initialize frame buffer.
        
        Args:
            capacity: Maximum frames to buffer
            target_fps: Target playback rate
        """
        self.capacity = capacity
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        
        self._frames: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        self._dropped_frames = 0
        self._total_frames = 0
    
    def put(
        self,
        frame_lines: List[str],
        timestamp: float,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """Add frame to buffer.
        
        Args:
            frame_lines: Rendered frame lines
            timestamp: Frame timestamp in seconds
            block: Whether to block if full
            timeout: Maximum wait time
            
        Returns:
            True if added, False if timed out
        """
        with self._not_full:
            if len(self._frames) >= self.capacity:
                if not block:
                    self._dropped_frames += 1
                    return False
                
                if not self._not_full.wait(timeout):
                    self._dropped_frames += 1
                    return False
            
            self._frames.append((frame_lines, timestamp))
            self._total_frames += 1
            self._not_empty.notify()
            return True
    
    def get(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[List[str], float]]:
        """Get frame from buffer.
        
        Args:
            block: Whether to block if empty
            timeout: Maximum wait time
            
        Returns:
            (frame_lines, timestamp) or None if timed out
        """
        with self._not_empty:
            if not self._frames:
                if not block:
                    return None
                
                if not self._not_empty.wait(timeout):
                    return None
            
            if self._frames:
                frame = self._frames.popleft()
                self._not_full.notify()
                return frame
            
            return None
    
    def peek(self) -> Optional[Tuple[List[str], float]]:
        """Peek at next frame without removing.
        
        Returns:
            Next frame or None if empty
        """
        with self._lock:
            return self._frames[0] if self._frames else None
    
    def clear(self) -> None:
        """Clear all buffered frames."""
        with self._lock:
            self._frames.clear()
            self._not_full.notify_all()
    
    @property
    def size(self) -> int:
        """Current number of buffered frames."""
        with self._lock:
            return len(self._frames)
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._frames) >= self.capacity
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._frames) == 0
    
    @property
    def dropped_frames(self) -> int:
        """Number of dropped frames."""
        return self._dropped_frames
    
    @property
    def fill_ratio(self) -> float:
        """Buffer fill ratio (0.0-1.0)."""
        with self._lock:
            return len(self._frames) / max(1, self.capacity)
    
    def wait_for_buffer(
        self,
        min_frames: Optional[int] = None,
        timeout: float = 10.0,
    ) -> bool:
        """Wait for buffer to fill to minimum level.
        
        Args:
            min_frames: Minimum frames to wait for (None=50% capacity)
            timeout: Maximum wait time in seconds
            
        Returns:
            True if reached minimum, False if timed out
        """
        if min_frames is None:
            min_frames = self.capacity // 2
        
        import time
        start = time.time()
        
        while self.size < min_frames:
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        
        return True
