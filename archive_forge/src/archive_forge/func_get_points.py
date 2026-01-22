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
def get_points(self):
    """
        Generates a list of all coordinate points within the grid.

        Returns:
            list of tuples: A list containing all (x, y) coordinates in the grid.
        """
    points = [(x, y) for x in range(WIDTH) for y in range(HEIGHT)]
    logging.debug(f'Generated {len(points)} points for the grid.')
    return points