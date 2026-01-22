"""
Module: search2048_overhauled.py
This module is part of the 2048 AI overhaul project, focusing on the game's search and decision-making mechanisms.
It includes the setup and main game loop, handling user input for tile movement, and comparing vector positions.

TODO:
- Integrate with AI decision-making components.
- Optimize performance for large search spaces.
- Expand functionality to support additional game features.
"""

# Importing necessary modules, classes, and libraries
from typing import Tuple, List, Optional
import numpy as np
from player_overhauled import (
    Player,
)  # Assuming Player class is defined in player_overhauled.py
import logging
import pygame  # Importing pygame for UI interactions

# Setting up logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initializing global variables
released: bool = True
teleport: bool = False

max_depth: int = 4  # Depth of search for AI decision making
pause_counter: int = 100
next_connection_no: int = 1000
speed: int = 60
move_speed: int = 60

x_offset: int = 0
y_offset: int = 0

# Placeholder for the Player instance
p: Optional[Player] = None


def setup() -> None:
    """
    Sets up the game environment, including frame rate and window size, and initializes the player.
    Utilizes pygame for creating the game window and setting up the environment.
    """
    pygame.init()
    screen = pygame.display.set_mode((850, 850))
    pygame.display.set_caption("2048 AI Overhaul")
    global p
    p = Player()
    logging.debug("Game setup completed using pygame.")


def draw(screen: pygame.Surface) -> None:
    """
    Main game loop responsible for drawing the game state on each frame.
    It updates the background, draws the grid, and handles tile movements.
    Utilizes pygame for drawing the game state.

    Parameters:
        screen (pygame.Surface): The pygame screen surface to draw the game state.
    """
    screen.fill((187, 173, 160))  # Setting background color
    for i in range(4):
        for j in range(4):
            pygame.draw.rect(
                screen,
                (205, 193, 180),
                (i * 200 + (i + 1) * 10, j * 200 + (j + 1) * 10, 200, 200),
            )
    pygame.display.update()

    if p and p.done_moving():
        p.get_move()
        p.move_tiles()


def key_pressed(event: pygame.event.Event) -> None:
    """
    Handles key press events to control tile movements using pygame.

    Parameters:
        event (pygame.event.Event): The pygame event object containing key press information.
    """
    global released
    if released:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if p and p.done_moving():
                    p.move_direction = np.array([0, -1])
                    p.move_tiles()
            elif event.key == pygame.K_DOWN:
                if p and p.done_moving():
                    p.move_direction = np.array([0, 1])
                    p.move_tiles()
            elif event.key == pygame.K_LEFT:
                if p and p.done_moving():
                    p.move_direction = np.array([-1, 0])
                    p.move_tiles()
            elif event.key == pygame.K_RIGHT:
                if p and p.done_moving():
                    p.move_direction = np.array([1, 0])
                    p.move_tiles()
        released = False


def key_released() -> None:
    """
    Resets the key release state to allow for new key press actions.
    This function is called when a key is released in the pygame event loop.
    """
    global released
    released = True


def compare_vec(p1: np.array, p2: np.array) -> bool:
    """
    Compares two vectors for equality.

    Parameters:
        p1 (np.array): The first vector.
        p2 (np.array): The second vector.

    Returns:
        bool: True if the vectors are equal, False otherwise.
    """
    return np.array_equal(p1, p2)
