import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
Returns an integer array of the same length as *scaled_positions*,
        tagging all equivalent atoms with the same index.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sg.tag_sites([[0.0, 0.0, 0.0],
        ...               [0.5, 0.5, 0.0],
        ...               [1.0, 0.0, 0.0],
        ...               [0.5, 0.0, 0.0]])
        array([0, 0, 0, 1])
        