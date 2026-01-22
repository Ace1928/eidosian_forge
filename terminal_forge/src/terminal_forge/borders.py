#!/usr/bin/env python3
"""
Borders

Description of the module's purpose and functionality following Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction

This module should embody core Terminal Forge architectural patterns while maintaining
crystal-clear intent and optimal performance characteristics.
🧱 Eidosian Border System 📏
Structured elegance for terminal artistry.

Form and function merge into perfect geometric harmony.
"""
from typing import Dict, List, Tuple, Optional, Union, Any

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Border Style Definitions - Structure as Control  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class BorderStyle:
    """
    Border style definitions with atomic precision.
    Each style is a complete mathematical system with perfect balance.
    """
    # Border component keys:
    # tl = top-left, tr = top-right, bl = bottom-left, br = bottom-right
    # h = horizontal, v = vertical
    # Extend with: tj = top junction, bj = bottom junction, 
    # lj = left junction, rj = right junction, x = crossing
    
    # Classical styles - timeless elegance
    ASCII = {
        'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
        'h': '-', 'v': '|',
        'tj': '+', 'bj': '+', 'lj': '+', 'rj': '+', 'x': '+'
    }
    
    SINGLE = {
        'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
        'h': '─', 'v': '│',
        'tj': '┬', 'bj': '┴', 'lj': '├', 'rj': '┤', 'x': '┼'
    }
    
    DOUBLE = {
        'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
        'h': '═', 'v': '║',
        'tj': '╦', 'bj': '╩', 'lj': '╠', 'rj': '╣', 'x': '╬'
    }
    
    ROUNDED = {
        'tl': '╭', 'tr': '╮', 'bl': '╰', 'br': '╯',
        'h': '─', 'v': '│',
        'tj': '┬', 'bj': '┴', 'lj': '├', 'rj': '┤', 'x': '┼'
    }
    
    BOLD = {
        'tl': '┏', 'tr': '┓', 'bl': '┗', 'br': '┛',
        'h': '━', 'v': '┃',
        'tj': '┳', 'bj': '┻', 'lj': '┣', 'rj': '┫', 'x': '╋'
    }
    
    # Aesthetic styles - artistic expression
    DIAMOND = {
        'tl': '◆', 'tr': '◆', 'bl': '◆', 'br': '◆',
        'h': '◆', 'v': '◆',
        'tj': '◆', 'bj': '◆', 'lj': '◆', 'rj': '◆', 'x': '◆'
    }
    
    STARS = {
        'tl': '✧', 'tr': '✧', 'bl': '✧', 'br': '✧',
        'h': '✦', 'v': '✧',
        'tj': '✧', 'bj': '✧', 'lj': '✧', 'rj': '✧', 'x': '✧'
    }
    
    BLOCKS = {
        'tl': '█', 'tr': '█', 'bl': '█', 'br': '█',
        'h': '█', 'v': '█',
        'tj': '█', 'bj': '█', 'lj': '█', 'rj': '█', 'x': '█'
    }
    
    DOTS = {
        'tl': '•', 'tr': '•', 'bl': '•', 'br': '•',
        'h': '·', 'v': '•',
        'tj': '•', 'bj': '•', 'lj': '•', 'rj': '•', 'x': '•'
    }
    
    SHADOW = {
        'tl': '▓', 'tr': '▓', 'bl': '░', 'br': '░',
        'h': '▓', 'v': '▓',
        'tj': '▓', 'bj': '░', 'lj': '▓', 'rj': '▓', 'x': '▓'
    }

    # System registry - dynamic style management
    _style_registry = {}

    @classmethod
    def get_style(cls, name: str) -> Dict[str, str]:
        """
        Retrieve border style by name with quantum reliability.
        If the style exists in any dimension, you'll get it.
        """
        # Check built-in styles
        try:
            return getattr(cls, name.upper())
        except AttributeError:
            # Check custom registered styles
            if name.lower() in cls._style_registry:
                return cls._style_registry[name.lower()]
            # Fallback with witty message
            print(f"Style '{name}' not found in this dimension—using ASCII fallback! 🌌")
            return cls.ASCII

    @classmethod
    def register_style(cls, name: str, style_dict: Dict[str, str]) -> None:
        """
        Register a new border style with universal persistence.
        Create your own geometric reality—the multiverse welcomes it.
        """
        required_keys = {'tl', 'tr', 'bl', 'br', 'h', 'v'}
        
        # Validate style components with quantum certainty
        if not all(k in style_dict for k in required_keys):
            missing = required_keys - set(style_dict.keys())
            raise ValueError(f"Incomplete border style! Missing components: {missing}. Even art needs structure! 🎭")
            
        # Register with automatic junction creation if missing
        full_style = style_dict.copy()
        
        # Add missing junction characters with intelligent defaults
        if 'tj' not in full_style: full_style['tj'] = style_dict['h']
        if 'bj' not in full_style: full_style['bj'] = style_dict['h']
        if 'lj' not in full_style: full_style['lj'] = style_dict['v']
        if 'rj' not in full_style: full_style['rj'] = style_dict['v']
        if 'x' not in full_style: full_style['x'] = style_dict['tl']
            
        # Commit to the style registry—permanent geometric reality
        cls._style_registry[name.lower()] = full_style
        print(f"Style '{name}' has materialized in the universe of borders! ✨")

    @classmethod
    def list_styles(cls) -> List[str]:
        """
        List all available border styles—complete dimensional awareness.
        Know your options before making a cosmic choice.
        """
        # Get built-in styles through reflection
        builtin_styles = [attr for attr in dir(cls) if 
                         attr.isupper() and 
                         isinstance(getattr(cls, attr), dict)]
        
        # Combine with registered styles for complete dimensional mapping
        all_styles = builtin_styles + list(cls._style_registry.keys())
        return sorted(all_styles)

    @classmethod
    def create_line(cls, style: Dict[str, str], width: int, line_type: str = 'h') -> str:
        """
        Create a horizontal or vertical line with specified style.
        Mathematical precision in one dimension.
        """
        if line_type == 'h':  # Horizontal line
            return style['h'] * width
        elif line_type == 'v':  # Vertical line
            return '\n'.join([style['v']] * width)
        else:
            raise ValueError(f"Line type '{line_type}' does not exist in this dimension! Choose 'h' or 'v'. 📏")

    @classmethod
    def create_box(cls, style: Dict[str, str], width: int, height: int) -> List[str]:
        """
        Create a complete box with specified style, width, and height.
        Perfect geometry manifested in terminal space.
        """
        if width < 2 or height < 2:
            raise ValueError("Box must be at least 2x2 in size. Even quarks have minimum dimensions! 🔬")
            
        box = []
        # Top border
        box.append(f"{style['tl']}{style['h'] * (width-2)}{style['tr']}")
        
        # Middle lines
        for _ in range(height - 2):
            box.append(f"{style['v']}{' ' * (width-2)}{style['v']}")
            
        # Bottom border
        box.append(f"{style['bl']}{style['h'] * (width-2)}{style['br']}")
        
        return box

# Self-testing capability—geometric self-awareness
if __name__ == "__main__":
    print("Available border styles:")
    for style_name in BorderStyle.list_styles():
        print(f"- {style_name}")
        
    # Show example of each built-in style
    styles_to_show = ["ASCII", "SINGLE", "DOUBLE", "ROUNDED", "BOLD"]
    for style_name in styles_to_show:
        style = BorderStyle.get_style(style_name)
        print(f"\n{style_name} style example:")
        box = BorderStyle.create_box(style, 20, 3)
        print('\n'.join(box))
        
    # Register a custom style
    print("\nRegistering custom style...")
    BorderStyle.register_style("hearts", {
        'tl': '❤', 'tr': '❤', 'bl': '❤', 'br': '❤',
        'h': '♥', 'v': '♥'
    })
    
    heart_style = BorderStyle.get_style("hearts")
    heart_box = BorderStyle.create_box(heart_style, 20, 3)
    print('\n'.join(heart_box))

