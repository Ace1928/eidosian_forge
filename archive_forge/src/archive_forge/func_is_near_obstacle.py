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
def is_near_obstacle(self, position, obstacle_positions=None):
    """
        Checks if a given position is near an obstacle on the grid.

        Args:
            position (tuple): The (x, y) position to check.
            obstacle_positions (list of tuples, optional): A list of obstacle positions to consider. If not provided, the method will use the grid's cells to determine obstacles.

        Returns:
            bool: True if the position is near an obstacle, False otherwise.
        """
    if obstacle_positions is None:
        obstacle_positions = [(x, y) for x in range(self.width) for y in range(self.height) if self.cells[y, x] != 0]
    for obstacle in obstacle_positions:
        if np.linalg.norm(np.array(position) - np.array(obstacle)) <= 1:
            logging.debug(f'Position {position} is near an obstacle at {obstacle}')
            return True
    logging.debug(f'Position {position} is not near any obstacles')
    return False