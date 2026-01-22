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
def get_lll_reduced_lattice(self, delta: float=0.75) -> Self:
    """Lenstra-Lenstra-Lovasz lattice basis reduction.

        Args:
            delta: Delta parameter.

        Returns:
            Lattice: LLL reduced
        """
    if delta not in self._lll_matrix_mappings:
        self._lll_matrix_mappings[delta] = self._calculate_lll()
    cls = type(self)
    return cls(self._lll_matrix_mappings[delta][0])