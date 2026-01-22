import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def broadcast_three(a: List[int], b: List[int], c: List[int]):
    return broadcast(broadcast(a, b), c)