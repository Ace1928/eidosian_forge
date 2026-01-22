import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
@staticmethod
def get_all_vertexes_from_edges(edges):
    """
        determines all vertex of a graph
        :param edges: list of edges
        :return: list of vertexes
        """
    vertexes = []
    for left, right in edges:
        if left not in vertexes:
            vertexes.append(left)
        if right not in vertexes:
            vertexes.append(right)
    return vertexes