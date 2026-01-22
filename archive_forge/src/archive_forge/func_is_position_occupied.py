import pygame as pg
import sys
from random import randint, seed
from collections import deque
from typing import List, Tuple, Deque, Optional, Set, Dict
import logging
import math
from queue import PriorityQueue
def is_position_occupied(self, position: Tuple[int, int]) -> bool:
    return position in self.snake_positions or position == self.fruit_position