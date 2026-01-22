import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def nullable_greater(left, right):
    if left is None or right is None:
        return False
    return left > right