import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Dict, Deque, Set, Optional, Callable
import logging
import math
from heapq import heappush, heappop
from itertools import product

# Setup logging configuration
logging.basicConfig(
    filename="snake_game_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Seed the random generator for reproducible results during development
seed(0)

# Initialize pygame
pg.init()
# Set the display to fullscreen at the current display resolution
display_info = pg.display.Info()  # Get current display information
SCREEN_WIDTH: int = display_info.current_w  # Set width to current width of display
SCREEN_HEIGHT: int = display_info.current_h  # Set height to current height of display
window: pg.Surface = pg.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT), pg.FULLSCREEN
)  # Set display mode to fullscreen
logging.info(
    f"Display set to fullscreen with resolution {SCREEN_WIDTH}x{SCREEN_HEIGHT}"
)
BLOCK_SIZE: int = 20
pg.display.set_caption("Enhanced Snake Game with A* Pathfinding")
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import multiprocessing
from queue import PriorityQueue
from typing import Set, List, Tuple, Dict, Deque
from collections import deque
import math

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
from collections import deque
from heapq import heappush, heappop


class Fruit:
    def __init__(self, window: pg.Surface) -> None:
        """
        Initializes a Fruit object with a position on the game window.
        This method sets the initial position of the fruit to a default value and then relocates it to a valid random position.
        """
        self.position: Tuple[int, int] = (0, 0)  # Initialize with a default position
        self.relocate()  # Relocate to a valid random position
        self.window = window
        logging.info(f"Fruit initialized and placed at {self.position}")

    def draw(self) -> None:
        """
        Draw the fruit on the game window using a fixed color (red) and block size.
        """
        color: Tuple[int, int, int] = (255, 0, 0)  # RGB color for the fruit
        pg.draw.rect(self.window, color, (*self.position, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Fruit drawn at {self.position} with color {color}")

    def relocate(self, exclude: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        Relocate the fruit to a random position within the game boundaries that is not occupied.
        Ensures the fruit does not spawn inside the snake's body or any other excluded positions.

        :param exclude: A list of positions to be excluded when placing the fruit.
        """
        if exclude is None:
            exclude = []  # Initialize exclude list if not provided
        valid_position_found: bool = False
        while not valid_position_found:
            new_x: int = randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_y: int = randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position: Tuple[int, int] = (new_x, new_y)
            if new_position not in exclude:
                self.position = new_position
                valid_position_found = True
                logging.info(f"Fruit relocated to {self.position}")
            else:
                logging.debug(
                    f"Attempted fruit position {new_position} is invalid, recalculating..."
                )


class Snake:
    def __init__(self, window: pg.Surface) -> None:
        """
        Initializes the Snake object with a starting position and a fruit object.
        """
        self.window = window
        self.body: Deque[Tuple[int, int]] = deque([(160, 160), (140, 160), (120, 160)])
        self.direction: Tuple[int, int] = (BLOCK_SIZE, 0)  # Moving right initially
        self.last_direction: Tuple[int, int] = (
            self.direction
        )  # Store the last direction
        self.fruit: Fruit = Fruit(window)
        self.path: List[Tuple[int, int]] = self.calculate_path()

    def calculate_path(self) -> List[Tuple[int, int]]:
        """
        Calculate the path from the snake's head to the fruit using the A* algorithm.
        This method ensures that the snake will attempt to find a path to the fruit. If no path is found,
        it will continue moving in the last valid direction to avoid stalling.
        """
        # Attempt to find a path using the A* algorithm
        path = a_star(
            start=self.body[0],  # Starting position is the head of the snake
            goal=self.fruit.position,  # Goal is the position of the fruit
            obstacles=set(self.body),  # The snake's body serves as obstacles
            path_so_far=list(self.body),  # Current path is the snake's body
            target=self.fruit.position,  # Target is again the position of the fruit
        )

        # If the A* algorithm fails to find a path, continue in the last known direction
        if not path:
            logging.debug("No path found using A*, continuing in last direction.")
            head_x, head_y = self.body[0]
            next_pos = (
                head_x
                + self.last_direction[
                    0
                ],  # Continue moving in the last horizontal direction
                head_y
                + self.last_direction[
                    1
                ],  # Continue moving in the last vertical direction
            )
            path = [
                next_pos
            ]  # The path is now just the next position in the last known direction

        # Log the calculated or continued path for debugging purposes
        logging.info(f"Path calculated or continued: {path}")
        return path

    def move(self) -> bool:
        """
        Move the snake based on the A* pathfinding result. Handles collision and game over scenarios.
        """
        if not self.path:
            self.path = self.calculate_path()  # Recalculate path if needed
        if self.path:
            next_pos: Tuple[int, int] = self.path.pop(0)
        else:
            # Continue in the last direction if no path is found
            head_x, head_y = self.body[0]
            next_pos = (
                head_x + self.last_direction[0],
                head_y + self.last_direction[1],
            )

        if self.is_collision(next_pos):
            logging.warning("Collision detected or snake out of bounds")
            return False

        self.body.appendleft(next_pos)
        self.last_direction = (
            next_pos[0] - self.body[1][0],
            next_pos[1] - self.body[1][1],
        )  # Update last direction

        if next_pos == self.fruit.position:
            self.fruit.relocate(list(self.body))
            self.path = self.calculate_path()  # Recalculate path after eating the fruit
        else:
            self.body.pop()

        logging.info(f"Snake moved successfully to {next_pos}")
        return True

    def draw(self) -> None:
        """
        Draw the snake on the game window using a fixed color (green) and block size for each segment.
        """
        color: Tuple[int, int, int] = (0, 255, 0)  # RGB color for the snake
        for segment in list(self.body):
            pg.draw.rect(self.window, color, (*segment, BLOCK_SIZE, BLOCK_SIZE))
        logging.debug(f"Snake drawn on window at segments: {list(self.body)}")

    def is_collision(self, position: Tuple[int, int]) -> bool:
        """
        Check if the given position results in a collision or is out of game bounds.
        Specifically checks if the head collides with the body or the boundaries.
        """
        # Check if the head collides with any part of the body except the last segment (tail)
        body_without_tail = list(self.body)[:-1]
        return (
            position in body_without_tail
            or not (0 <= position[0] < SCREEN_WIDTH)
            or not (0 <= position[1] < SCREEN_HEIGHT)
        )


def main():
    clock = pg.time.Clock()
    snake = Snake(window)
    fruit = Fruit(window)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        window.fill((0, 0, 0))
        snake.fruit.draw()
        snake.draw()

        if not snake.move():
            logging.info("Game Over, restarting")
            print("Game Over")
            snake = Snake(window)  # Restart the game

        pg.display.flip()
        clock.tick(60)
        logging.debug("Game loop executed")


if __name__ == "__main__":
    main()
