from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations

        Get the valid neighboring positions of a given position.

        This function takes a position and returns a list of its valid neighboring positions.
        The neighboring positions are calculated by adding the four cardinal directions (up, right, down, left) to the current position.
        The validity of each neighboring position is checked against the grid boundaries and obstacle positions.
        Only the positions that are within the grid boundaries and not occupied by obstacles are considered valid neighbors.

        Args:
            position (Tuple[int, int]): The position for which to get the neighbors.

        Returns:
            List[Tuple[int, int]]: A list of valid neighboring positions.
        