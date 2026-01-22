from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def calculate_body_position_proximity(self, position: Tuple[int, int], body_positions: Set[Tuple[int, int]], space_around_agent: int) -> float:
    """
        Calculate a penalty for being too close to the snake's own body.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            body_positions: A set of positions occupied by the snake's body as tuples of (x, y) coordinates.
            space_around_agent: The desired space to maintain around the snake's body.

        Returns:
            The calculated penalty for being too close to the snake's body.
        """
    penalty = 0.0
    for body_pos in body_positions:
        if self.calculate_distance(position, body_pos) < space_around_agent:
            penalty += float('inf')
    return penalty