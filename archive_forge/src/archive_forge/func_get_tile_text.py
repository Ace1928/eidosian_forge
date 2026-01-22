from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def get_tile_text(value: int) -> str:
    """
    Generates the text to display on the tile based on its value.

    Args:
        value (int): The value of the tile.

    Returns:
        str: The text to display on the tile.
    """
    if value == 0:
        return ''
    return str(value)