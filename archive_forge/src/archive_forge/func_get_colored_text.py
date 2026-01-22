from typing import Dict, List, Optional, TextIO
def get_colored_text(text: str, color: str) -> str:
    """Get colored text."""
    color_str = _TEXT_COLOR_MAPPING[color]
    return f'\x1b[{color_str}m\x1b[1;3m{text}\x1b[0m'