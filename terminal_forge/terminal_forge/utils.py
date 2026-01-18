#!/usr/bin/env python3
"""
Utils

Description of the module's purpose and functionality following Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction

This module should embody core Terminal Forge architectural patterns while maintaining
crystal-clear intent and optimal performance characteristics.

🛠️ Eidosian Banner Utilities 🧰
Precision tools that execute with atomic efficiency.

Zero waste. Maximum impact. Pure utility.
"""
import re
import shutil
import time
from typing import Tuple, List, Dict, Any, Callable, Optional

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Terminal & Text Manipulation Utilities    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def get_terminal_size() -> Tuple[int, int]:
    """
    Quantum-detect terminal dimensions with fallback intelligence.
    Never fails—adapts like water to any container.
    """
    try:
        return shutil.get_terminal_size()
    except (AttributeError, OSError):
        return (80, 24)  # Universal constants in the terminal multiverse

def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences with regex precision.
    Like a quantum filter that removes color without disturbing content.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def text_length(text: str) -> int:
    """
    Calculate visual length of text, ignoring invisible ANSI codes.
    What you see is what you measure—quantum observer principle in action.
    """
    return len(strip_ansi(text))

def center_text(text: str, width: int) -> str:
    """
    Center text with perfect mathematical balance.
    The center of gravity for your string—Newton would be proud.
    """
    text_width = text_length(text)
    if text_width >= width:
        return text  # Already wider than container—conservation of space
    padding = (width - text_width) // 2  # Integer division ensures symmetry
    return " " * padding + text

def right_align(text: str, width: int) -> str:
    """
    Right-align text with perfect precision.
    Like gravity pulling your text to the right edge of the universe.
    """
    text_width = text_length(text)
    if text_width >= width:
        return text
    return " " * (width - text_width) + text

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text while preserving ANSI formatting—surgical precision.
    Like a quantum scissors that cuts content but preserves the essence.
    """
    if text_length(text) <= max_length:
        return text
    
    # Find cutoff point that respects ANSI sequences—dimensional awareness
    result = ""
    current_length = 0
    i = 0
    
    while current_length < max_length - len(suffix) and i < len(text):
        if text[i] == '\033':  # ANSI escape sequence detected
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
    
    # Add suffix and reset any unclosed color codes
    return result + suffix + '\033[0m'

def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator that measures function execution time—quantum precision.
    Time is the ultimate currency; spend it wisely.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"⏱️ {func.__name__} executed in {elapsed:.6f}s")
        return result
    return wrapper

def split_text_into_lines(text: str, max_width: int) -> List[str]:
    """
    Split text into lines of maximum width with smart word wrapping.
    Words flow like water, finding their natural line breaks.
    """
    if not text:
        return []
        
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        word_len = text_length(word)
        if current_width + word_len + len(current_line) > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_len
        else:
            current_line.append(word)
            current_width += word_len
            
    if current_line:  # Add the last line if not empty
        lines.append(" ".join(current_line))
        
    return lines

def create_progress_bar(progress: float, width: int = 20, fill_char: str = "█", empty_char: str = "░") -> str:
    """
    Generate a beautiful progress bar with precise proportions.
    Progress visualized as a slice of the universe, one character at a time.
    """
    if not 0 <= progress <= 1:
        raise ValueError("Progress must be between 0 and 1. Even quantum particles have limits! 🔬")
        
    filled_length = int(width * progress)
    bar = fill_char * filled_length + empty_char * (width - filled_length)
    percentage = int(progress * 100)
    return f"{bar} {percentage}%"

# Self-test module functionality—introspection is intelligence
if __name__ == "__main__":
    print("Terminal size:", get_terminal_size())
    print("Centered text:", center_text("Hello Universe", 30))
    print("Text length:", text_length("\033[31mColored\033[0m Text"))
    print("Progress bar:", create_progress_bar(0.75))
