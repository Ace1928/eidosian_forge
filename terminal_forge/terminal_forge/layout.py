#!/usr/bin/env python3
"""
Layout

Description of the module's purpose and functionality following Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction

This module should embody core Terminal Forge architectural patterns while maintaining
crystal-clear intent and optimal performance characteristics.

📏 Eidosian Layout Engine 🧩
Quantum-precise spatial arrangement for terminal art.

Where mathematics meets aesthetics—spatial harmony manifested in code.
"""
from enum import Enum
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from dataclasses import dataclass

# Local imports with elegant error handling
try:
    from .utils import strip_ansi, text_length, truncate_text
except ImportError:
    # Graceful fallback for direct module execution
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from terminal_forge.utils import strip_ansi, text_length, truncate_text

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 📐 Layout Core - Spatial Harmony in Terminal     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

class Alignment(Enum):
    """Text alignment options with perfect balance."""
    LEFT = "left"     # ◀️ Left alignment - text starts at left edge
    CENTER = "center" # ⚫ Center alignment - text centered in space
    RIGHT = "right"   # ▶️ Right alignment - text ends at right edge
    JUSTIFY = "justify" # ⬌ Justified - text spans full width

@dataclass
class Margin:
    """
    Margin settings for perfect spacing around elements.
    Like the space around a painting—breathing room for visual impact.
    """
    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0
    
    @classmethod
    def uniform(cls, value: int) -> 'Margin':
        """Create uniform margins on all sides."""
        return cls(value, value, value, value)
    
    @classmethod
    def horizontal(cls, value: int) -> 'Margin':
        """Create horizontal margins (left and right)."""
        return cls(0, value, 0, value)
    
    @classmethod
    def vertical(cls, value: int) -> 'Margin':
        """Create vertical margins (top and bottom)."""
        return cls(value, 0, value, 0)

@dataclass
class Padding:
    """
    Padding settings for perfect spacing within elements.
    Like cushioning inside a box—intimate space that enhances content.
    """
    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0
    
    @classmethod
    def uniform(cls, value: int) -> 'Padding':
        """Create uniform padding on all sides."""
        return cls(value, value, value, value)
    
    @classmethod
    def horizontal(cls, value: int) -> 'Padding':
        """Create horizontal padding (left and right)."""
        return cls(0, value, 0, value)
    
    @classmethod
    def vertical(cls, value: int) -> 'Padding':
        """Create vertical padding (top and bottom)."""
        return cls(value, 0, value, 0)

class ContentOverflow(Enum):
    """
    Content overflow handling strategies.
    How text behaves when it exceeds its container—like water finding its path.
    """
    WRAP = "wrap"       # ↩️ Wrap to next line
    TRUNCATE = "truncate" # ✂️ Cut off with ellipsis
    SCROLL = "scroll"   # ⏩ Allow content to extend (scroll)
    ERROR = "error"     # 🚫 Raise error on overflow

class LayoutEngine:
    """
    Layout engine for precise positioning of banner elements.
    Universal mathematics applied to visual design—dimensions in perfect harmony.
    """
    
    @staticmethod
    def align_text(text: str, width: int, alignment: Alignment = Alignment.LEFT) -> str:
        """
        Align text within a given width based on the specified alignment.
        Mathematical precision in text positioning—balanced to the atomic level.
        """
        text_width = text_length(text)
        
        # If text already exceeds width, return unchanged
        if text_width >= width:
            return text
            
        # Handle different alignments with mathematical precision
        if alignment == Alignment.CENTER:
            padding = (width - text_width) // 2  # Integer division for perfect balance
            return " " * padding + text + " " * (width - text_width - padding)
        elif alignment == Alignment.RIGHT:
            return " " * (width - text_width) + text
        elif alignment == Alignment.JUSTIFY and " " in text:
            # Only justify if there are spaces to expand
            words = text.split(" ")
            if len(words) <= 1:  # Cannot justify single word
                return text.ljust(width)
                
            # Calculate space distribution
            spaces_needed = width - sum(text_length(word) for word in words)
            spaces_between_words = len(words) - 1
            space_width = spaces_needed // spaces_between_words
            extra_spaces = spaces_needed % spaces_between_words
            
            # Build justified text with mathematical precision
            result = ""
            for i, word in enumerate(words[:-1]):  # All but last word
                result += word
                space_count = space_width + (1 if i < extra_spaces else 0)
                result += " " * space_count
            result += words[-1]  # Add last word
            return result
        else:  # LEFT alignment or fallback
            return text + " " * (width - text_width)
    
    @staticmethod
    def apply_padding(text_lines: List[str], padding: Padding, width: int) -> List[str]:
        """
        Apply padding to text lines with pixel-perfect precision.
        Like the precise spacing in a well-designed book—each margin intentional.
        """
        padded_lines = []
        
        # Add top padding
        for _ in range(padding.top):
            padded_lines.append(" " * width)
            
        # Add horizontal padding to each content line
        for line in text_lines:
            padded_line = " " * padding.left + line + " " * padding.right
            padded_lines.append(padded_line)
            
        # Add bottom padding
        for _ in range(padding.bottom):
            padded_lines.append(" " * width)
            
        return padded_lines
    
    @staticmethod
    def handle_overflow(text: str, max_width: int, strategy: ContentOverflow = ContentOverflow.WRAP) -> List[str]:
        """
        Handle text overflow using the specified strategy.
        Like a river finding its path—text flows according to natural principles.
        """
        if not text:
            return [""]
            
        # No overflow, return as single line
        if text_length(text) <= max_width:
            return [text]
            
        # Apply overflow strategy with quantum precision
        if strategy == ContentOverflow.TRUNCATE:
            return [truncate_text(text, max_width)]
        elif strategy == ContentOverflow.SCROLL:
            return [text]  # Allow text to extend beyond width
        elif strategy == ContentOverflow.ERROR:
            raise ValueError(f"Text exceeds max width of {max_width}! Trim your cosmic message or choose a different overflow strategy! 📏✂️")
        else:  # Default to WRAP
            return LayoutEngine.wrap_text(text, max_width)
    
    @staticmethod
    def wrap_text(text: str, max_width: int) -> List[str]:
        """
        Wrap text to fit within max_width with optimal word boundaries.
        Like perfectly folded origami—each line break positioned for maximum elegance.
        """
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_len = text_length(word)
            
            # If adding word exceeds max_width, start a new line
            if current_width + word_len + len(current_line) > max_width:
                if current_line:  # Avoid empty lines
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_len
            else:
                current_line.append(word)
                current_width += word_len
                
        # Add the last line if not empty
        if current_line:
            lines.append(" ".join(current_line))
            
        return lines
    
    @staticmethod
    def create_columns(content_lists: List[List[str]], widths: List[int], 
                      alignments: List[Alignment] = None, separator: str = " ") -> List[str]:
        """
        Create multi-column layout with precise alignment.
        Like the perfect columns of an ancient temple—mathematics made visible.
        """
        if not content_lists:
            return []
            
        # Validate inputs with quantum precision
        if len(widths) != len(content_lists):
            raise ValueError("Column count mismatch! Each content list needs its own width dimension! 🏛️")
            
        # Set default alignments if not provided
        if not alignments:
            alignments = [Alignment.LEFT] * len(content_lists)
        elif len(alignments) != len(content_lists):
            raise ValueError("Alignment count mismatch! Each column needs its own alignment! ⬆️⬇️")
            
        # Find the maximum number of rows
        max_rows = max(len(col) for col in content_lists)
        
        # Build row by row with mathematical harmony
        result = []
        for row_idx in range(max_rows):
            row_parts = []
            
            for col_idx, content_list in enumerate(content_lists):
                # Get content or empty string if row doesn't exist
                content = content_list[row_idx] if row_idx < len(content_list) else ""
                # Apply alignment with cosmic precision
                aligned = LayoutEngine.align_text(content, widths[col_idx], alignments[col_idx])
                row_parts.append(aligned)
                
            # Join columns with separator
            result.append(separator.join(row_parts))
            
        return result
    
    @staticmethod
    def create_table(headers: List[str], rows: List[List[str]], widths: List[int] = None, 
                    alignments: List[Alignment] = None, border_style: Dict[str, str] = None) -> List[str]:
        """
        Create a table with headers, rows, and borders.
        Like the periodic table—information arranged with crystalline structure.
        """
        # Default border style if not provided
        if not border_style:
            from .borders import BorderStyle
            border_style = BorderStyle.SINGLE
            
        # Calculate column widths if not provided
        if not widths:
            # Find maximum width needed for each column
            col_count = len(headers)
            widths = [0] * col_count
            
            # Check headers
            for col_idx, header in enumerate(headers):
                widths[col_idx] = max(widths[col_idx], text_length(header))
                
            # Check all rows
            for row in rows:
                for col_idx, cell in enumerate(row[:col_count]):  # Limit to header count
                    widths[col_idx] = max(widths[col_idx], text_length(cell))
                    
            # Add padding for readability
            widths = [w + 2 for w in widths]
        
        # Set default alignments if not provided
        if not alignments:
            alignments = [Alignment.CENTER] * len(headers)
            
        # Build the table with architectural precision
        result = []
        
        # Top border
        top_border = border_style['tl']
        for width in widths:
            top_border += border_style['h'] * width + border_style['tj']
        result.append(top_border[:-1] + border_style['tr'])  # Replace last junction with corner
        
        # Headers row with alignment
        header_row = border_style['v']
        for col_idx, header in enumerate(headers):
            aligned = LayoutEngine.align_text(header, widths[col_idx], alignments[col_idx])
            header_row += aligned + border_style['v']
        result.append(header_row)
        
        # Separator after headers
        separator = border_style['lj']
        for width in widths:
            separator += border_style['h'] * width + border_style['x']
        result.append(separator[:-1] + border_style['rj'])  # Replace last junction with right junction
        
        # Data rows
        for row in rows:
            data_row = border_style['v']
            for col_idx, cell in enumerate(row[:len(headers)]):  # Limit to header count
                aligned = LayoutEngine.align_text(cell, widths[col_idx], alignments[col_idx])
                data_row += aligned + border_style['v']
            result.append(data_row)
            
        # Bottom border
        bottom_border = border_style['bl']
        for width in widths:
            bottom_border += border_style['h'] * width + border_style['bj']
        result.append(bottom_border[:-1] + border_style['br'])  # Replace last junction with corner
        
        return result
    
    @staticmethod
    def responsive_width(content: str, min_width: int = 10, max_width: int = 120, 
                       padding_ratio: float = 0.1) -> int:
        """
        Calculate responsive width based on content and constraints.
        Like a responsive website—adapting perfectly to its environment.
        """
        # Find base width from content
        content_width = max(text_length(line) for line in content.split("\n"))
        
        # Apply padding ratio for aesthetic spacing
        padded_width = int(content_width * (1 + 2 * padding_ratio))  # Padding on both sides
        
        # Constrain within min and max bounds
        width = max(min_width, min(padded_width, max_width))
        
        # Ensure width is even for symmetry
        return width + (width % 2)

# Self-awareness test function
if __name__ == "__main__":
    print("📏 Testing Eidosian Layout Engine...")
    
    # Test text alignment
    test_text = "Cosmic alignment"
    width = 30
    print(f"\n🔍 Testing text alignment (width: {width}):")
    for align in Alignment:
        aligned = LayoutEngine.align_text(test_text, width, align)
        print(f"{align.value.title():8}: '{aligned}'")
    
    # Test multi-column layout
    print("\n🔍 Testing column layout:")
    col1 = ["Alpha", "Beta", "Gamma"]
    col2 = ["First", "Second"]
    col3 = ["1", "2", "3", "4"]
    
    columns = LayoutEngine.create_columns(
        [col1, col2, col3],
        [10, 10, 5],
        [Alignment.LEFT, Alignment.CENTER, Alignment.RIGHT]
    )
    
    for line in columns:
        print(f"'{line}'")
    
    # Test text wrapping
    long_text = "The Eidosian Layout Engine applies mathematical precision to text formatting, creating visual harmony in terminal space."
    print("\n🔍 Testing text wrapping (width: 30):")
    wrapped = LayoutEngine.wrap_text(long_text, 30)
    for line in wrapped:
        print(f"'{line}'")
    
    # Test table creation
    print("\n🔍 Testing table creation:")
    headers = ["Name", "Type", "Value"]
    rows = [
        ["Alpha", "Greek", "1"],
        ["Beta", "Greek", "2"],
        ["Gamma", "Greek", "3"]
    ]
    
    table = LayoutEngine.create_table(headers, rows)
    for line in table:
        print(line)
    
    print("\n✅ Layout engine tests complete!")
