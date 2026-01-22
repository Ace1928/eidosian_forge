from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def calculate_block_size(screen_width: int, screen_height: int) -> int:
    """
    Calculate the block size based on the screen resolution.

    This function calculates the block size dynamically based on the screen resolution
    to ensure visibility and proportionality. It takes the screen width and height as
    input and returns the calculated block size as an integer.

    Args:
        screen_width (int): The width of the screen in pixels.
        screen_height (int): The height of the screen in pixels.

    Returns:
        int: The calculated block size.
    """
    reference_resolution = (1920, 1080)
    reference_block_size = 20
    scaling_factor = min(screen_width / reference_resolution[0], screen_height / reference_resolution[1])
    dynamic_block_size = max(1, int(reference_block_size * scaling_factor))
    adjusted_block_size = min(max(dynamic_block_size, 1), 30)
    return adjusted_block_size