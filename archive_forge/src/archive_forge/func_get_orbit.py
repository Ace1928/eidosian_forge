from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Literal, overload
import numpy as np
from monty.design_patterns import cached_class
from monty.serialization import loadfn
from pymatgen.util.string import Stringify
def get_orbit(self, p: ArrayLike, tol: float=1e-05) -> list[np.ndarray]:
    """Returns the orbit for a point.

        Args:
            p: Point as a 3x1 array.
            tol: Tolerance for determining if sites are the same. 1e-5 should
                be sufficient for most purposes. Set to 0 for exact matching
                (and also needed for symbolic orbits).

        Returns:
            list[array]: Orbit for point.
        """
    orbit: list[np.ndarray] = []
    for o in self.symmetry_ops:
        pp = o.operate(p)
        pp = np.mod(np.round(pp, decimals=10), 1)
        if not in_array_list(orbit, pp, tol=tol):
            orbit.append(pp)
    return orbit