"""
Color utilities.
"""
import colorsys
from typing import List, Tuple

def generate_hsv_palette(n: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors
