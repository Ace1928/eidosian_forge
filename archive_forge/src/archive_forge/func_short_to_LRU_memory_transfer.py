import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def short_to_LRU_memory_transfer(short_term_memory: ShortTermMemory, long_term_memory: LRUMemory) -> None:
    """
    Transfers relevant information from short-term memory to long-term memory for learning and optimisation.

    Args:
        short_term_memory (ShortTermMemory): The short-term memory instance.
        LRU_memory (LRUMemory): The long-term memory instance.
    """
    while short_term_memory.memory:
        board, move, score = short_term_memory.memory.popleft()
        long_term_memory.store(board, move, score)