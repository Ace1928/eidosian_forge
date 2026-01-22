from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
@property
def reciprocal_lattice_crystallographic(self) -> Self:
    """Returns the *crystallographic* reciprocal lattice, i.e. no factor of 2 * pi."""
    cls = type(self)
    return cls(self.reciprocal_lattice.matrix / (2 * np.pi))