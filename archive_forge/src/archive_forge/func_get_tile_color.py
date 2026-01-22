from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def get_tile_color(value: int) -> tuple:
    """
    Generates a color for the tile based on its value, using a gradient approach.

    Args:
        value (int): The value of the tile.

    Returns:
        tuple: The color (R, G, B) for the tile.
    """
    if value == 0:
        return (205, 193, 180)
    base_log = log2(value)
    base_color = (255 - min(int(base_log * 20), 255), 255 - min(int(base_log * 15), 255), 220)
    return base_color