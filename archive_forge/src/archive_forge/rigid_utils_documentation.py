from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

        Moves the transformation object to GPU memory

        Returns:
            A version of the transformation on GPU
        