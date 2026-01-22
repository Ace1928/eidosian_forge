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
@classmethod
def rhombohedral(cls, a: float, alpha: float, pbc: PbcLike=(True, True, True)) -> Self:
    """Convenience constructor for a rhombohedral lattice.

        Args:
            a (float): *a* lattice parameter of the rhombohedral cell.
            alpha (float): Angle for the rhombohedral lattice in degrees.
            pbc (tuple): a tuple defining the periodic boundary conditions along the three
                axis of the lattice. If None periodic in all directions.

        Returns:
            Rhombohedral lattice of dimensions a x a x a.
        """
    return cls.from_parameters(a, a, a, alpha, alpha, alpha, pbc=pbc)