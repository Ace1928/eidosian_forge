import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math
from queue import PriorityQueue
def restart_game(self) -> None:
    """Restarts the game following a collision or similar event."""
    self.body = deque([(20, 20), (40, 20), (60, 20)])
    self.score = 0
    self.growing = 0
    self.grid.update_snake_position(self.body)
    self.fruit.relocate()
    logging.info('Game restarted.')