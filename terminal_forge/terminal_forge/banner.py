#!/usr/bin/env python3
"""
Terminal Forge Banner System

A modular, stylized banner creation system for terminal applications following Eidosian principles:
- Contextual Integrity: Every component serves a precise purpose in the visual communication system
- Precision as Style: Form follows function with minimal waste and maximum clarity
- Structure as Control: Organized architecture enables predictable, composable banner creation
- Flow Like a River: Operations chain naturally with a fluent interface pattern

This module provides a complete banner generation system for CLI applications with features including:
- Customizable borders, colors, and text alignment
- Theme system for consistent visual styling
- Special effects like animation and blinking
- Color utilities including gradients and rainbow text
- Factory methods for common banner types (info, warning, success, error)

Classes:
    Color: Terminal color system with RGB and style support
    BorderStyle: Collection of border styles for banner customization
    Alignment: Enum for text alignment options (LEFT, CENTER, RIGHT)
    Banner: Core banner generation and rendering class
    Theme: Predefined color schemes for consistent styling
    Effect: Special visual effects for banners
    BannerFactory: Shortcuts for creating common banner types

Usage:
    >>> from terminal_forge import banner
    >>> banner.Banner("Welcome").add_line("Hello, world!").display()
    >>> themed = banner.Theme.apply(banner.Banner("Warning"), "warning")
    >>> themed.add_line("Caution required").display()
"""
import os
import sys
import time
import random
import re
import shutil
from enum import Enum
from typing import List, Dict, Callable, Union, Optional, Tuple, Any
from functools import lru_cache, wraps

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🌈 Terminal Color System - Precise yet Universal  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class Color:
    """Color system with automatic terminal capability detection."""
    RESET = "\033[0m"
    # Base colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright variants
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    # Styles
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"

    @staticmethod
    @lru_cache(maxsize=32)
    def rgb(r: int, g: int, b: int) -> str:
        """Get RGB color code - cached for performance."""
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    @lru_cache(maxsize=32)
    def bg_rgb(r: int, g: int, b: int) -> str:
        """Get RGB background color code - cached for performance."""
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def rainbow(text: str) -> str:
        """🌈 Apply rainbow coloring to text."""
        colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.CYAN, Color.BLUE, Color.MAGENTA]
        result = ""
        for i, char in enumerate(text):
            if char.strip():  # Only color non-whitespace
                result += f"{colors[i % len(colors)]}{char}{Color.RESET}"
            else:
                result += char
        return result

    @staticmethod
    def gradient(text: str, start_color: Tuple[int, int, int], end_color: Tuple[int, int, int]) -> str:
        """Apply a smooth gradient to text."""
        if not text:
            return ""
        
        result = ""
        for i, char in enumerate(text):
            if char.strip():  # Only color non-whitespace
                # Calculate color at this position
                r = int(start_color[0] + (end_color[0] - start_color[0]) * i / len(text))
                g = int(start_color[1] + (end_color[1] - start_color[1]) * i / len(text))
                b = int(start_color[2] + (end_color[2] - start_color[2]) * i / len(text))
                result += f"{Color.rgb(r, g, b)}{char}{Color.RESET}"
            else:
                result += char
        return result

    @staticmethod
    def supports_color() -> bool:
        """Check if the terminal supports color."""
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            'COLORTERM' in os.environ or 'TERM' in os.environ and 
            os.environ.get('TERM') != 'dumb'
        )

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🧱 Border Styles - Structure with Personality    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class BorderStyle:
    """Border styles for banner boxes."""
    # Simple borders
    ASCII = {
        'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
        'h': '-', 'v': '|'
    }
    
    SINGLE = {
        'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
        'h': '─', 'v': '│'
    }
    
    DOUBLE = {
        'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
        'h': '═', 'v': '║'
    }
    
    ROUNDED = {
        'tl': '╭', 'tr': '╮', 'bl': '╰', 'br': '╯',
        'h': '─', 'v': '│'
    }
    
    BOLD = {
        'tl': '┏', 'tr': '┓', 'bl': '┗', 'br': '┛',
        'h': '━', 'v': '┃'
    }
    
    DIAMOND = {
        'tl': '◆', 'tr': '◆', 'bl': '◆', 'br': '◆',
        'h': '◆', 'v': '◆'
    }
    
    STARS = {
        'tl': '✧', 'tr': '✧', 'bl': '✧', 'br': '✧',
        'h': '✦', 'v': '✧'
    }

    @classmethod
    def get_style(cls, name: str) -> Dict[str, str]:
        """Get border style by name or return ASCII if not found."""
        try:
            return getattr(cls, name.upper())
        except AttributeError:
            return cls.ASCII

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🔧 Utility Functions - Small but Mighty Helpers  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def get_terminal_size() -> Tuple[int, int]:
    """Get terminal size or sensible defaults if not available."""
    try:
        return shutil.get_terminal_size()
    except (AttributeError, OSError):
        return (80, 24)  # Sensible fallback

def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def text_length(text: str) -> int:
    """Get actual display length of text by stripping ANSI sequences."""
    return len(strip_ansi(text))

def center_text(text: str, width: int) -> str:
    """Center text considering actual display width."""
    text_width = text_length(text)
    if text_width >= width:
        return text
    padding = (width - text_width) // 2
    return " " * padding + text

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, considering ANSI sequences."""
    if text_length(text) <= max_length:
        return text
    
    # Find the cutoff point that respects ANSI sequences
    result = ""
    current_length = 0
    i = 0
    
    while current_length < max_length - len(suffix) and i < len(text):
        if text[i] == '\033':  # ANSI escape sequence
            j = i
            while j < len(text) and text[j] != 'm':
                j += 1
            if j < len(text):
                result += text[i:j+1]
                i = j + 1
                continue
        
        result += text[i]
        current_length += 1
        i += 1
    
    return result + suffix + Color.RESET

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🎁 Banner Class - Core Implementation            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class Alignment(Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class Banner:
    """
    Modular banner generator with styling and layout options.
    Perfect for CLI applications seeking a dash of style.
    """
    def __init__(self, title: str = "", width: int = 0):
        """Initialize a new banner with optional title and width."""
        self.title = title
        self.content_lines = []
        self.width = width or get_terminal_size()[0] - 4  # Default with some margin
        self.padding = 1
        self.border_style = BorderStyle.SINGLE
        self.border_color = None
        self.title_color = None
        self.content_color = None
        self.alignment = Alignment.LEFT
    
    def set_border(self, style_name: str) -> 'Banner':
        """Set border style by name."""
        self.border_style = BorderStyle.get_style(style_name)
        return self
    
    def set_border_color(self, color: str) -> 'Banner':
        """Set border color."""
        self.border_color = color
        return self
    
    def set_title_color(self, color: str) -> 'Banner':
        """Set title color."""
        self.title_color = color
        return self
    
    def set_content_color(self, color: str) -> 'Banner':
        """Set content color."""
        self.content_color = color
        return self
    
    def set_alignment(self, alignment: Union[Alignment, str]) -> 'Banner':
        """Set text alignment."""
        if isinstance(alignment, str):
            alignment = Alignment(alignment.lower())
        self.alignment = alignment
        return self
    
    def set_padding(self, padding: int) -> 'Banner':
        """Set padding inside the banner."""
        self.padding = max(0, padding)
        return self
    
    def add_line(self, line: str) -> 'Banner':
        """Add a line of content to the banner."""
        self.content_lines.append(line)
        return self
    
    def add_lines(self, lines: List[str]) -> 'Banner':
        """Add multiple lines of content to the banner."""
        self.content_lines.extend(lines)
        return self
    
    def add_separator(self, char: str = "─") -> 'Banner':
        """Add a separator line to the banner."""
        self.content_lines.append(char * (self.width - 2 * self.padding))
        return self
    
    def _apply_border_color(self, text: str) -> str:
        """Apply border color if set."""
        if self.border_color:
            return f"{self.border_color}{text}{Color.RESET}"
        return text
    
    def _apply_title_color(self, text: str) -> str:
        """Apply title color if set."""
        if self.title_color:
            return f"{self.title_color}{text}{Color.RESET}"
        return text
    
    def _apply_content_color(self, text: str) -> str:
        """Apply content color if set."""
        if self.content_color:
            return f"{self.content_color}{text}{Color.RESET}"
        return text
    
    def _align_text(self, text: str, width: int) -> str:
        """Align text according to the alignment setting."""
        text_width = text_length(text)
        
        if self.alignment == Alignment.CENTER:
            return center_text(text, width)
        elif self.alignment == Alignment.RIGHT:
            if text_width >= width:
                return text
            return " " * (width - text_width) + text
        else:  # LEFT
            return text
    
    def _format_line(self, line: str) -> str:
        """Format a content line with proper alignment and width."""
        max_width = self.width - (2 * self.padding + 2)  # Account for borders and padding
        if text_length(line) > max_width:
            line = truncate_text(line, max_width)
        
        padding_str = " " * self.padding
        bordered_width = self.width - 2  # Width inside borders
        
        aligned_text = self._align_text(line, bordered_width - 2 * self.padding)
        colored_text = self._apply_content_color(aligned_text)
        
        v_border = self._apply_border_color(self.border_style['v'])
        
        return f"{v_border}{padding_str}{colored_text}{' ' * (bordered_width - text_length(aligned_text) - 2 * self.padding)}{padding_str}{v_border}"
    
    def render(self) -> str:
        """Render the banner to a string."""
        result = []
        
        # Top border
        top_border = (
            self._apply_border_color(self.border_style['tl']) + 
            self._apply_border_color(self.border_style['h'] * (self.width - 2)) + 
            self._apply_border_color(self.border_style['tr'])
        )
        result.append(top_border)
        
        # Title if present
        if self.title:
            title_text = f" {self.title} "
            if text_length(title_text) > self.width - 4:
                title_text = truncate_text(title_text, self.width - 4)
            
            centered_title = center_text(title_text, self.width - 2)
            title_line = (
                self._apply_border_color(self.border_style['v']) + 
                self._apply_title_color(centered_title) + 
                self._apply_border_color(self.border_style['v'])
            )
            result.append(title_line)
            
            # Separator after title
            separator = (
                self._apply_border_color(self.border_style['v']) + 
                self._apply_border_color(self.border_style['h'] * (self.width - 2)) + 
                self._apply_border_color(self.border_style['v'])
            )
            result.append(separator)
        
        # Empty line if no content
        if not self.content_lines:
            empty_line = (
                self._apply_border_color(self.border_style['v']) + 
                " " * (self.width - 2) + 
                self._apply_border_color(self.border_style['v'])
            )
            result.append(empty_line)
        
        # Content lines
        for line in self.content_lines:
            result.append(self._format_line(line))
        
        # Bottom border
        bottom_border = (
            self._apply_border_color(self.border_style['bl']) + 
            self._apply_border_color(self.border_style['h'] * (self.width - 2)) + 
            self._apply_border_color(self.border_style['br'])
        )
        result.append(bottom_border)
        
        return "\n".join(result)
    
    def display(self) -> None:
        """Display the banner to stdout."""
        print(self.render())

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🎨 Theme System - Consistent Visual Language     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class Theme:
    """Banner theme with predefined color schemes."""
    
    THEMES = {
        "default": {
            "border": Color.BRIGHT_WHITE,
            "title": Color.BOLD + Color.BRIGHT_WHITE,
            "content": Color.WHITE,
        },
        "error": {
            "border": Color.RED,
            "title": Color.BOLD + Color.BRIGHT_RED,
            "content": Color.WHITE,
        },
        "success": {
            "border": Color.GREEN,
            "title": Color.BOLD + Color.BRIGHT_GREEN,
            "content": Color.WHITE,
        },
        "warning": {
            "border": Color.YELLOW,
            "title": Color.BOLD + Color.BRIGHT_YELLOW,
            "content": Color.WHITE,
        },
        "info": {
            "border": Color.BLUE,
            "title": Color.BOLD + Color.BRIGHT_BLUE,
            "content": Color.WHITE,
        },
        "cyberpunk": {
            "border": Color.rgb(255, 0, 153),
            "title": Color.BOLD + Color.rgb(0, 255, 255),
            "content": Color.rgb(204, 204, 0),
        },
        "matrix": {
            "border": Color.BRIGHT_GREEN,
            "title": Color.BOLD + Color.GREEN,
            "content": Color.GREEN,
        },
        "ocean": {
            "border": Color.rgb(0, 153, 255),
            "title": Color.BOLD + Color.rgb(0, 204, 255),
            "content": Color.CYAN,
        }
    }
    
    @classmethod
    def apply(cls, banner: Banner, theme_name: str) -> Banner:
        """Apply a theme to a banner."""
        if theme_name not in cls.THEMES:
            raise ValueError(f"Theme '{theme_name}' not found. Available themes: {', '.join(cls.THEMES.keys())}")
            
        theme = cls.THEMES[theme_name]
        return banner.set_border_color(theme["border"]) \
                    .set_title_color(theme["title"]) \
                    .set_content_color(theme["content"])
    
    @classmethod
    def register_theme(cls, name: str, border_color: str, title_color: str, content_color: str) -> None:
        """Register a new theme."""
        cls.THEMES[name] = {
            "border": border_color,
            "title": title_color,
            "content": content_color
        }

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🌟 Special Effects - Adding Flair with Purpose   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class Effect:
    """Special effects for banners."""
    
    @staticmethod
    def animate_typing(banner: Banner, delay: float = 0.05) -> None:
        """Animate banner with typing effect."""
        full_text = banner.render()
        lines = full_text.split('\n')
        
        # Clear the display area
        print('\n' * len(lines))
        sys.stdout.write(f"\033[{len(lines)}A")
        sys.stdout.flush()
        
        # Display each character with delay
        displayed = ["" for _ in range(len(lines))]
        for i in range(max(len(line) for line in lines)):
            for j, line in enumerate(lines):
                if i < len(line):
                    displayed[j] += line[i]
                    sys.stdout.write(f"\033[{j+1};1H{displayed[j]}")
            sys.stdout.flush()
            time.sleep(delay)
        print()  # Final newline
    
    @staticmethod
    def blink(banner: Banner, blink_times: int = 3, delay: float = 0.3) -> None:
        """Make the banner blink."""
        full_text = banner.render()
        for _ in range(blink_times * 2):
            # Clear the area
            lines = full_text.count('\n') + 1
            print('\n' * lines)
            sys.stdout.write(f"\033[{lines}A")
            
            # Toggle visibility
            if _ % 2 == 0:
                print(full_text)
            else:
                print('\n' * (lines - 1))
            
            sys.stdout.flush()
            time.sleep(delay)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🏭 Banner Factory - Quick Creation Shortcuts     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class BannerFactory:
    """Factory for quickly creating common banner types."""
    
    @staticmethod
    def info(message: str, title: str = "INFO") -> Banner:
        """Create an information banner."""
        banner = Banner(title)
        banner.add_line(message)
        return Theme.apply(banner, "info")
    
    @staticmethod
    def success(message: str, title: str = "SUCCESS") -> Banner:
        """Create a success banner."""
        banner = Banner(title)
        banner.add_line(message)
        return Theme.apply(banner, "success")
    
    @staticmethod
    def warning(message: str, title: str = "WARNING") -> Banner:
        """Create a warning banner."""
        banner = Banner(title)
        banner.add_line(message)
        return Theme.apply(banner, "warning")
    
    @staticmethod
    def error(message: str, title: str = "ERROR") -> Banner:
        """Create an error banner."""
        banner = Banner(title)
        banner.add_line(message)
        return Theme.apply(banner, "error")
    
    @staticmethod
    def ascii_art(art: str, title: str = "") -> Banner:
        """Create a banner with ASCII art."""
        banner = Banner(title)
        for line in art.split('\n'):
            banner.add_line(line)
        return banner

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🔍 Diagnostic & Self-Testing - Continuous Check  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def self_test() -> None:
    """Run self-tests for the banner system."""
    print("🔍 Running Eidosian Banner self-tests...")
    
    # Test basic banner
    print("\n📋 Testing basic banner:")
    Banner("Hello World").add_line("This is a test banner.").display()
    
    # Test themed banner
    print("\n🎨 Testing themed banner:")
    Theme.apply(Banner("Success"), "success").add_line("Operation completed successfully!").display()
    
    # Test border styles
    print("\n🧱 Testing border styles:")
    for style in ["single", "double", "rounded", "bold"]:
        Banner(f"{style.title()} Border").set_border(style).add_line("Custom border style").display()
    
    # Test text alignment
    print("\n📐 Testing text alignment:")
    for alignment in [Alignment.LEFT, Alignment.CENTER, Alignment.RIGHT]:
        Banner(f"{alignment.value.title()} Alignment").set_alignment(alignment).add_line("Text alignment demo").display()
    
    # Test color effects
    print("\n🌈 Testing color effects:")
    Banner("Rainbow Text").add_line(Color.rainbow("This text has rainbow colors applied!")).display()
    
    start_color = (255, 0, 0)  # Red
    end_color = (0, 0, 255)    # Blue
    Banner("Gradient Text").add_line(Color.gradient("This text transitions from red to blue!", start_color, end_color)).display()
    
    print("\n✅ All tests completed successfully!")

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🚀 Quick Examples - See It In Action            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def example_usage() -> None:
    """Demonstrate example usage of the banner system."""
    # Basic usage
    Banner("Welcome to Eidosian Banner Forge") \
        .add_line("A modular, stylish banner creation system.") \
        .add_line("Perfect for CLI applications seeking a dash of style.") \
        .display()
    
    # Themed banner with special border
    Theme.apply(Banner("✨ Cyberpunk Theme ✨"), "cyberpunk") \
        .set_border("double") \
        .add_line("Neon lights illuminate the digital horizon.") \
        .add_line("Hack the planet! 🌐") \
        .display()
    
    # Warning message
    BannerFactory.warning("System resources are running low!", "⚠️ RESOURCE ALERT").display()
    
    # ASCII art banner
    art = r"""
   /\_/\  
  ( o.o ) 
   > ^ <
    """
    BannerFactory.ascii_art(art, "Eidosian Cat").display()
    
    # Animated banner (commented out as it might not be desired in all contexts)
    # banner = Banner("Animated Text")
    # banner.add_line("This banner appears with a typing effect!")
    # Effect.animate_typing(banner, delay=0.03)

# Main execution guard
if __name__ == "__main__":
    # Run self-tests and examples if script is executed directly
    self_test()
    print("\n" + "=" * 60 + "\n")
    example_usage()
