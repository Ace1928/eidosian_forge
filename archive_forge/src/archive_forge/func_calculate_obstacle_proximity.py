from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def calculate_obstacle_proximity(self, position: Tuple[int, int], obstacles: Set[Tuple[int, int]], space_around_obstacles: int) -> float:
    """
        Calculate a penalty based on the proximity to obstacles.

        Args:
            position: The current position as a tuple of (x, y) coordinates.
            obstacles: A set of obstacle positions as tuples of (x, y) coordinates.
            space_around_obstacles: The desired space to maintain around obstacles.

        Returns:
            The calculated penalty based on proximity to obstacles.
        """
    penalty = 0.0
    for obstacle in obstacles:
        distance = self.calculate_distance(position, obstacle)
        if distance <= space_around_obstacles:
            penalty += 1 / (distance + 1)
    return penalty