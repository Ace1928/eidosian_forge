from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def calculate_obstacle_proximity_penalty(self, position: Tuple[int, int], space_around_obstacles: int=5) -> float:
    """
        Calculate a penalty score based on the proximity to the nearest obstacle.

        This function iterates through each obstacle within the line of sight and calculates the distance to the given position.
        If the distance is less than the specified space around obstacles, a penalty is calculated based on the inverse of the distance.
        Closer obstacles are assigned a higher penalty to emphasize their significance.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            space_around_obstacles (int): The minimum desired distance from any obstacle. Default is 5.

        Returns:
            float: The total penalty accumulated from all nearby obstacles within the line of sight.
        """
    penalty: float = 0.0
    visible_obstacles: Set[Tuple[int, int]] = self.get_line_of_sight_obstacles(position)
    for obstacle in visible_obstacles:
        distance: float = self.calculate_euclidean_distance(position, obstacle)
        if distance <= space_around_obstacles:
            penalty += 1 / (distance + 1)
    return penalty