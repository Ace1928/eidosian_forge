import pygame
import random
import heapq
import logging
from typing import List, Optional, Dict, Any, Tuple
import cProfile
from collections import deque
import numpy as np
import time
import torch
from functools import lru_cache as LRUCache
import math
import asyncio
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from queue import PriorityQueue
from collections import defaultdict
def is_valid_position(self, position):
    """
        Determines whether a given position is valid for the amoeba-inspired Hamiltonian pathfinding algorithm.
        This method thoroughly checks the position against the grid boundaries, obstacles, and the snake's body,
        ensuring that only accessible and safe cells are considered as valid steps in the path.

        Args:
            position (tuple): The position to validate.

        Returns:
            bool: True if the position is valid and accessible, False otherwise.
        """
    x, y = position
    if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
        return False
    if position in self.snake.segments:
        return False
    if self.grid.is_obstacle(position):
        return False
    return True