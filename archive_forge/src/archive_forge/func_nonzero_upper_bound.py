import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def nonzero_upper_bound(input: List[int]):
    return [numel(input), len(input)]