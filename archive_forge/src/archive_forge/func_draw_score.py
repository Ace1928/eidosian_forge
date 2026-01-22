from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
from math import log2
import pygame
@StandardDecorator()
def draw_score(score: int) -> None:
    """
    Draws the current score to the GUI for visualization.

    Args:
        score (int): The current score.
    """