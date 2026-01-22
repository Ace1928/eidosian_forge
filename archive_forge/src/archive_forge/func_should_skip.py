import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def should_skip(tensor: List[int]):
    return numel(tensor) == 0 and len(tensor) == 1