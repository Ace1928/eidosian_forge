import unittest
import math
from A_STAR import A_STAR
from pygame.math import Vector2
from typing import List, Tuple, Any, Set
from Utility import Grid, Node
from Snake import Snake
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
def test_select_optimal_path(self):
    path = [Node(1, 1), Node(1, 2), Node(1, 3), Node(1, 4)]
    optimal_path = self.a_star.select_optimal_path()
    self.assertEqual(optimal_path, path)