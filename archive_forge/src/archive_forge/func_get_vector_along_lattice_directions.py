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
def get_vector_along_lattice_directions(self, cart_coords: ArrayLike) -> np.ndarray:
    """Returns the coordinates along lattice directions given Cartesian coordinates.

        Note, this is different than a projection of the Cartesian vector along the
        lattice parameters. It is simply the fractional coordinates multiplied by the
        lattice vector magnitudes.

        For example, this method is helpful when analyzing the dipole moment (in
        units of electron Angstroms) of a ferroelectric crystal. See the `Polarization`
        class in `pymatgen.analysis.ferroelectricity.polarization`.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Lattice coordinates.
        """
    return self.lengths * self.get_fractional_coords(cart_coords)