from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
def process_parsed_coords(coords):
    """Takes a set of parsed coordinates, which come as an array of strings,
    and returns a numpy array of floats.
    """
    geometry = np.zeros(shape=(len(coords), 3), dtype=float)
    for ii, entry in enumerate(coords):
        for jj in range(3):
            geometry[ii, jj] = float(entry[jj])
    return geometry