import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
Prints a summary for a single layer (including topological connections).

    Args:
        layer: target layer.
    