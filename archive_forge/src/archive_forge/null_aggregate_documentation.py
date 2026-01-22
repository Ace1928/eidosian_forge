from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U

    Wrap finalizer with null handling.

    If the accumulation is empty or None, the returned finalizer returns None.

    Args:
        finalize: The core finalizing function to wrap.

    Returns:
        A new finalizing function that handles nulls.
    