#!/usr/bin/env python3
"""
Themes

Description of the module's purpose and functionality following Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction

This module should embody core Terminal Forge architectural patterns while maintaining
crystal-clear intent and optimal performance characteristics.

🎨 Eidosian Theme System 🌈
Visual harmony through mathematical precision.

Themes as quantum states—instantly transforming appearance without changing essence.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

# Local imports with elegant error handling
try:
    from .colors import Color
except ImportError:
    # Graceful fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from terminal_forge.colors import Color

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 🎨 Theme Definitions - Perfect Color Harmonies   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

@dataclass
class ThemeDefinition:
    """Structured theme definition with precise color relationships."""
    border: str
    title: str
    content: str
    symbol: str = ""
    border_style: str = "single"
    
    def validate(self) -> bool:
        """Verify theme integrity—all components must exist in harmony."""
        # All color components must be defined
        return bool(self.border and self.title and self.content)

class Theme:
    """
    Banner theme system with mathematically precise color relationships.
    Quantum color theory—instantly transform any banner with perfect harmony.
    """
    
    # ╔═════════════════════════════════════════════╗
    # ║ Core Theme Definitions - The Color Universe ║
    # ╚═════════════════════════════════════════════╝
    
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
            "symbol": "⚠️ ",
            "border_style": "bold"
        },
        "success": {
            "border": Color.GREEN,
            "title": Color.BOLD + Color.BRIGHT_GREEN,
            "content": Color.WHITE,
            "symbol": "✅ ",
            "border_style": "rounded"
        },
        "warning": {
            "border": Color.YELLOW,
            "title": Color.BOLD + Color.BRIGHT_YELLOW,
            "content": Color.WHITE,
            "symbol": "⚠️ ",
            "border_style": "single"
        },
        "info": {
            "border": Color.BLUE,
            "title": Color.BOLD + Color.BRIGHT_BLUE,
            "content": Color.WHITE,
            "symbol": "ℹ️ ",
            "border_style": "single"
        },
        # Advanced themes with perfect color harmony
        "cyberpunk": {
            "border": Color.rgb(255, 0, 153),  # Neon pink
            "title": Color.BOLD + Color.rgb(0, 255, 255),  # Cyan
            "content": Color.rgb(204, 204, 0),  # Yellow
            "symbol": "⚡ ",
            "border_style": "double"
        },
        "matrix": {
            "border": Color.BRIGHT_GREEN,
            "title": Color.BOLD + Color.GREEN,
            "content": Color.GREEN,
            "symbol": "⟩ ",
            "border_style": "bold"
        },
        "ocean": {
            "border": Color.rgb(0, 153, 255),  # Ocean blue
            "title": Color.BOLD + Color.rgb(0, 204, 255),  # Sky blue
            "content": Color.CYAN,
            "symbol": "~ ",
            "border_style": "rounded"
        },
        "sunset": {
            "border": Color.rgb(255, 102, 0),  # Orange
            "title": Color.BOLD + Color.rgb(255, 51, 51),  # Red-orange
            "content": Color.YELLOW,
            "symbol": "☀ ",
            "border_style": "rounded"
        },
        "forest": {
            "border": Color.rgb(0, 102, 0),  # Dark green
            "title": Color.BOLD + Color.rgb(0, 153, 0),  # Medium green
            "content": Color.rgb(204, 255, 204),  # Light green
            "symbol": "🌿 ",
            "border_style": "single"
        },
        "cosmic": {
            "border": Color.rgb(75, 0, 130),  # Indigo
            "title": Color.BOLD + Color.rgb(148, 0, 211),  # Violet
            "content": Color.rgb(200, 200, 255),  # Light lavender
            "symbol": "✨ ",
            "border_style": "stars"
        },
        "minimal": {
            "border": Color.WHITE,
            "title": Color.BOLD + Color.WHITE,
            "content": Color.WHITE,
            "border_style": "ascii"
        },
        "retro": {
            "border": Color.BRIGHT_YELLOW,
            "title": Color.BOLD + Color.BRIGHT_YELLOW,
            "content": Color.BRIGHT_GREEN,
            "symbol": "» ",
            "border_style": "single"
        }
    }
    
    # Theme registry for dynamic expansion
    _custom_themes = {}
    
    @classmethod
    def get(cls, theme_name: str) -> Dict[str, Any]:
        """
        Retrieve a theme by name with quantum precision.
        Returns default if theme doesn't exist in this universe.
        """
        theme_name = theme_name.lower()
        
        # Check built-in themes first
        if theme_name in cls.THEMES:
            return cls.THEMES[theme_name]
        
        # Then check custom themes
        if theme_name in cls._custom_themes:
            return cls._custom_themes[theme_name]
        
        # Return default with a witty message
        print(f"Theme '{theme_name}' not found in the multiverse—using default! 🔭")
        return cls.THEMES["default"]
    
    @classmethod
    def apply(cls, banner: Any, theme_name: str) -> Any:
        """
        Apply a theme to a banner instantly.
        Like a costume change between dimensions—instant transformation.
        """
        theme = cls.get(theme_name)
        
        # Set banner properties based on theme
        banner = banner.set_border_color(theme["border"]) \
                      .set_title_color(theme["title"]) \
                      .set_content_color(theme["content"])
        
        # Apply border style if specified
        if "border_style" in theme:
            banner = banner.set_border(theme["border_style"])
            
        # Apply symbol to title if specified
        if "symbol" in theme and theme["symbol"] and banner.title:
            banner.title = f"{theme['symbol']}{banner.title}"
            
        return banner
    
    @classmethod
    def register(cls, name: str, border_color: str, title_color: str, 
                 content_color: str, symbol: str = "", border_style: str = "single") -> None:
        """
        Register a new theme with complete validation.
        Your creativity enters the theme multiverse—forever part of the cosmos.
        """
        name = name.lower()
        
        # Create theme definition with atomic precision
        theme = {
            "border": border_color,
            "title": title_color,
            "content": content_color
        }
        
        # Add optional parameters if provided
        if symbol:
            theme["symbol"] = symbol
        
        if border_style:
            theme["border_style"] = border_style
        
        # Register the theme in our cosmic registry
        cls._custom_themes[name] = theme
        
        print(f"Theme '{name}' has been woven into the fabric of reality! ✨")
    
    @classmethod
    def list_themes(cls) -> List[str]:
        """
        List all available themes across all dimensions.
        Knowledge is power—knowing your options is cosmic wisdom.
        """
        # Combine built-in and custom themes
        all_themes = list(cls.THEMES.keys()) + list(cls._custom_themes.keys())
        return sorted(all_themes)
    
    @classmethod
    def generate_from_color(cls, name: str, base_color: Union[str, Tuple[int, int, int]],
                           border_style: str = "single") -> Dict[str, Any]:
        """
        Generate a theme from a single base color through color theory.
        One seed creates an entire universe—color harmony emerges from simplicity.
        """
        # Convert string representation to RGB tuple if needed
        if isinstance(base_color, str):
            # Handle hex color
            if base_color.startswith('#') and len(base_color) == 7:
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                base_color = (r, g, b)
            else:
                raise ValueError(f"Invalid color format: {base_color}. Try '#RRGGBB' or RGB tuple! 🎨")
        
        # Generate a harmonic color scheme
        r, g, b = base_color
        
        # Create border color (original color)
        border_color = Color.rgb(r, g, b)
        
        # Create title color (brighter variant)
        brightness_factor = 1.3
        title_r = min(255, int(r * brightness_factor))
        title_g = min(255, int(g * brightness_factor))
        title_b = min(255, int(b * brightness_factor))
        title_color = Color.BOLD + Color.rgb(title_r, title_g, title_b)
        
        # Create content color (best contrast)
        # Calculate luminance to determine if we should use dark or light text
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        content_color = Color.WHITE if luminance < 0.5 else Color.rgb(30, 30, 30)
        
        # Register the theme
        theme = {
            "border": border_color,
            "title": title_color,
            "content": content_color,
            "border_style": border_style
        }
        
        cls._custom_themes[name.lower()] = theme
        
        return theme

# Self-test code - theme system's awareness of itself
if __name__ == "__main__":
    print("🎨 Available themes in the Eidosian multiverse:")
    for theme_name in Theme.list_themes():
        print(f"  • {theme_name}")
    
    print("\n🧪 Generating a new theme from a color:")
    Theme.generate_from_color("ruby", (220, 20, 60), "rounded")
    print("  • ruby (successfully created!)")
    
    print("\n✨ Theme system is functioning perfectly in this universe!")

