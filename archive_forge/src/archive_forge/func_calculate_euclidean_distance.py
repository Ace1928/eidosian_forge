from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def calculate_euclidean_distance(self, position1: Tuple[int, int], position2: Tuple[int, int]) -> float:
    """
        Calculate the Euclidean distance between two positions in the grid.

        This function utilizes basic mathematical operations to calculate the Euclidean distance between two positions.
        The Euclidean distance is more suitable for grid-based pathfinding compared to Manhattan distance in certain scenarios.

        Args:
            position1 (Tuple[int, int]): The first position tuple.
            position2 (Tuple[int, int]): The second position tuple.

        Returns:
            float: The Euclidean distance between the two positions.
        """
    dx = position2[0] - position1[0]
    dy = position2[1] - position1[1]
    return math.sqrt(dx * dx + dy * dy)