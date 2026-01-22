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
def get_orbit_and_generators(self, p: ArrayLike, tol: float=1e-05) -> tuple[list[np.ndarray], list[SymmOp]]:
    """Returns the orbit and its generators for a point.

        Args:
            p: Point as a 3x1 array.
            tol: Tolerance for determining if sites are the same. 1e-5 should
                be sufficient for most purposes. Set to 0 for exact matching
                (and also needed for symbolic orbits).

        Returns:
            tuple[list[np.ndarray], list[SymmOp]]: Orbit and generators for point.
        """
    from pymatgen.core.operations import SymmOp
    orbit: list[np.ndarray] = [np.array(p, dtype=float)]
    identity = SymmOp.from_rotation_and_translation(np.eye(3), np.zeros(3))
    generators: list[np.ndarray] = [identity]
    for o in self.symmetry_ops:
        pp = o.operate(p)
        pp = np.mod(np.round(pp, decimals=10), 1)
        if not in_array_list(orbit, pp, tol=tol):
            orbit.append(pp)
            generators.append(o)
    return (orbit, generators)