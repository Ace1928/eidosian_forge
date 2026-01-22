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
def hamiltonian_cycle(graph: nx.Graph) -> np.ndarray:
    """
    Compute a Hamiltonian cycle in a graph using a refined Nearest Neighbor heuristic, enhanced with numpy for optimal performance.

    This function meticulously constructs a Hamiltonian cycle by initially selecting a random vertex and iteratively connecting the nearest unvisited neighbor based on edge weights, until all vertices are visited. It then returns to the starting vertex to complete the cycle.

    Args:
        graph (nx.Graph): The graph within which to find the Hamiltonian cycle.

    Returns:
        np.ndarray: A numpy array representing the Hamiltonian cycle as a sequence of vertices.

    Detailed Description:
        - The function initializes an empty list to store the cycle and a set to track visited vertices.
        - It selects a random starting vertex and marks it as visited.
        - Utilizing a while loop, the function continues to find the nearest unvisited neighbor based on the edge weight until all vertices are visited.
        - Upon visiting all vertices, the cycle is closed by returning to the starting vertex.
        - The cycle list is then converted to a numpy array for efficient data handling and returned.
    """
    cycle = np.empty((0, 2), dtype=int)
    visited = set()
    current_vertex = random.choice(list(graph.nodes))
    visited.add(current_vertex)
    cycle = np.append(cycle, [current_vertex], axis=0)
    while len(visited) < len(graph.nodes):
        neighbors = np.array(list(graph[current_vertex].keys()))
        mask = np.isin(neighbors, list(visited), invert=True)
        unvisited_neighbors = neighbors[mask]
        if unvisited_neighbors.size > 0:
            next_vertex = min(unvisited_neighbors, key=lambda n: graph[current_vertex][n]['weight'])
        else:
            next_vertex = min(neighbors, key=lambda n: graph[current_vertex][n]['weight'])
        cycle = np.append(cycle, [next_vertex], axis=0)
        visited.add(next_vertex)
        current_vertex = next_vertex
    cycle = np.append(cycle, [cycle[0]], axis=0)
    return cycle