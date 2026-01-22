from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_game_over() -> None:
    """
    Draws the game over screen to the GUI for visualization.
    """