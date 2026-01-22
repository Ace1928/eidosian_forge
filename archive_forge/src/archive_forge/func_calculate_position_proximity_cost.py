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
def calculate_position_proximity_cost(self, position, target_positions):
    """
        Calculates the cost associated with the proximity of a given position to a set of target positions.
        This method assesses the strategic value of a position based on its distance from important target locations.

        Args:
            position (tuple): The position to evaluate.
            target_positions (list): A list of target positions to consider.

        Returns:
            float: The proximity cost of the position relative to the target positions.
        """
    min_distance = min((np.linalg.norm(np.array(position) - np.array(target)) for target in target_positions))
    proximity_cost = 10 / (min_distance + 1)
    logging.debug(f'Calculated proximity cost for position {position}: {proximity_cost}')
    return proximity_cost