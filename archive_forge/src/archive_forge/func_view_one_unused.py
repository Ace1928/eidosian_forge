import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def view_one_unused(self: List[int], sizes: List[int], *, implicit: bool=False):
    return view(self, sizes)