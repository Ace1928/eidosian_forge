from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_game_pause() -> None:
    """
    Draws the game pause screen to the GUI for visualization.
    """