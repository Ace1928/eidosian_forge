import apple
from search import SearchAlgorithm
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def move_up(self) -> None:
    self.change_direction(0, -1)