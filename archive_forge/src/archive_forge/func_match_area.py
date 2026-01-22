from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
@property
def match_area(self):
    """The area of the match between the substrate and film super lattice vectors."""
    return vec_area(*self.film_sl_vectors)