import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def squeeze_nodim(li: List[int]):
    out: List[int] = []
    for i in range(len(li)):
        if li[i] != 1:
            out.append(li[i])
    return out