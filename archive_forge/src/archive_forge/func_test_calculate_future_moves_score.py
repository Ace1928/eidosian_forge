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
def test_calculate_future_moves_score(self):
    path = [Node(1, 1), Node(1, 2), Node(1, 3), Node(1, 4)]
    future_moves_score = self.a_star.calculate_future_moves_score(path)
    self.assertEqual(future_moves_score, 3)