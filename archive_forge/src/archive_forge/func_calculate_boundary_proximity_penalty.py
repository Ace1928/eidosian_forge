from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def calculate_boundary_proximity_penalty(self, position: Tuple[int, int], boundaries: Tuple[int, int, int, int]=(0, 0, 100, 100), space_around_boundaries: int=5) -> float:
    """
        Calculate a penalty based on the proximity to boundaries.

        This function computes a penalty score based on how close the given position is to the boundaries of the environment.
        The penalty increases as the position approaches the boundary within a specified threshold.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            boundaries (Tuple[int, int, int, int]): The boundaries of the environment as a tuple of (x_min, y_min, x_max, y_max). Default is (0, 0, 100, 100).
            space_around_boundaries (int): The desired space to maintain around boundaries. Default is 5.

        Returns:
            float: The calculated penalty based on proximity to boundaries.
        """
    x_min, y_min, x_max, y_max = boundaries
    min_distance_to_boundary: float = min(position[0] - x_min, x_max - position[0], position[1] - y_min, y_max - position[1])
    if min_distance_to_boundary < space_around_boundaries:
        return (space_around_boundaries - min_distance_to_boundary) ** 2
    return 0.0