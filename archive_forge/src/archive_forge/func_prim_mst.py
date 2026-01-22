import pygame  # Importing the pygame library to handle game-specific functionalities, providing a set of Python modules designed for writing video games.
import random  # Importing the random library to facilitate random number generation, crucial for unpredictability in game mechanics.
import heapq  # Importing the heapq library to provide an implementation of the heap queue algorithm, essential for efficient priority queue operations.
import logging  # Importing the logging library to enable logging of messages of varying severity, which is fundamental for tracking events that happen during runtime and for debugging.
import numpy as np  # Importing the numpy library as np to provide support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays, enhancing numerical computations.
import networkx as nx  # Importing the networkx library as nx to create, manipulate, and study the structure, dynamics, and functions of complex networks, useful for graph-based operations in computational models.
from collections import (
from typing import (
from functools import (
@lru_cache(maxsize=None)
def prim_mst(graph: nx.Graph, start_vertex: Tuple[int, int]) -> np.ndarray:
    """
    Calculate the Minimum Spanning Tree (MST) using Prim's algorithm starting from a specified vertex, utilizing advanced data structures and numpy for optimal performance.

    Args:
        graph (nx.Graph): The graph on which to perform the MST calculation, represented as a NetworkX graph object.
        start_vertex (Tuple[int, int]): The starting vertex for the MST, represented as a tuple of integers indicating the grid position.

    Returns:
        np.ndarray: A numpy array representing the MST as a sequence of vertices, ensuring high performance and efficient memory usage.

    Detailed Description:
        - The function initializes a numpy array for the MST and a set for tracking visited vertices.
        - It converts the adjacency list of the starting vertex into a numpy structured array for efficient heap operations.
        - Utilizes a min-heap to always extend the MST with the minimum weight edge to an unvisited vertex.
        - Iteratively processes the heap until all possible vertices are visited and included in the MST.
        - The numpy array structure allows for efficient append operations and memory management.
    """
    mst = np.empty((0, 2), dtype=int)
    visited = set([start_vertex])
    edges = np.array([(weight, start_vertex, to) for to, weight in graph[start_vertex].items()], dtype=[('weight', float), ('from', 'O'), ('to', 'O')])
    heapq.heapify(edges)
    while edges.size > 0:
        edge = heapq.heappop(edges)
        weight, frm, to = (edge['weight'], edge['from'], edge['to'])
        if to not in visited:
            visited.add(to)
            mst = np.append(mst, np.array([to], dtype=int), axis=0)
            for next_to, weight in graph[to].items():
                if next_to not in visited:
                    heapq.heappush(edges, (weight, to, next_to))
    return mst