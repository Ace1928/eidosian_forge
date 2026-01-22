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
def long_term_memory_to_short_term_transfer(long_term_memory: LongTermMemory, short_term_memory: ShortTermMemory) -> None:
    """
    Transfers relevant information from long-term memory to short-term memory for immediate decision-making.

    Args:
        long_term_memory (LongTermMemory): The long-term memory instance.
        short_term_memory (ShortTermMemory): The short-term memory instance.
    """
    for key, value in long_term_memory.memory.items():
        board, move, score = value
        short_term_memory.store(board, move, score)
        del long_term_memory.memory[key]