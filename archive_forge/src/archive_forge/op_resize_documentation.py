from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
Pad `data` in 'edge' mode, and get n nearest elements in the padded array and their indexes in the original array.

    Args:
        x: Center index (in the unpadded coordinate system) of the found
            nearest elements.
        n: The number of neighbors.
        data: The array.

    Returns:
        A tuple containing the indexes of neighbor elements (the index
        can be smaller than 0 or higher than len(data)) and the value of
        these elements.
    