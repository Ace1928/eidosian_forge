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
def is_constrained_edge(self, p1, p2):
    """
        Determines whether the edge between two points is constrained by an obstacle.
        This method performs a meticulous check to verify if the line segment connecting the given points
        intersects with any of the specified obstacle positions, ensuring the integrity of the constrained
        Delaunay triangulation.

        Args:
            p1 (tuple): The first point of the edge.
            p2 (tuple): The second point of the edge.

        Returns:
            bool: True if the edge is constrained by an obstacle, False otherwise.
        """
    for obstacle in self.obstacles:
        if self.is_point_on_line_segment(obstacle, p1, p2):
            logging.debug(f'Edge between {p1} and {p2} is constrained by obstacle at {obstacle}.')
            return True
    return False