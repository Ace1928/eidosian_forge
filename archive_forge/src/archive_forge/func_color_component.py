from typing import Dict, List
def color_component(x: int) -> int:
    """
    Implements the 6x6x6 color cube values of 8bit mode described at
    https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    """
    if x == 0:
        return 0
    return 55 + 40 * x