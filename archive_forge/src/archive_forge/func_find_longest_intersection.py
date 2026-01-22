import functools
from dataclasses import asdict, dataclass, field, replace
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from outlines.base import vectorize
from outlines.caching import cache
def find_longest_intersection(response: List[int], choice: List[int]) -> List[int]:
    """Find the longest intersection between the response and the choice."""
    for i, (token_r, token_c) in enumerate(zip_longest(response, choice)):
        if token_r != token_c:
            return response[:i]
    return response