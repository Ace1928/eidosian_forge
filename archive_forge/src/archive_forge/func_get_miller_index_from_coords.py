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
def get_miller_index_from_coords(self, coords: ArrayLike, coords_are_cartesian: bool=True, round_dp: int=4, verbose: bool=True) -> tuple[int, int, int]:
    """Get the Miller index of a plane from a list of site coordinates.

        A minimum of 3 sets of coordinates are required. If more than 3 sets of
        coordinates are given, the best plane that minimises the distance to all
        points will be calculated.

        Args:
            coords (iterable): A list or numpy array of coordinates. Can be
                Cartesian or fractional coordinates. If more than three sets of
                coordinates are provided, the best plane that minimises the
                distance to all sites will be calculated.
            coords_are_cartesian (bool, optional): Whether the coordinates are
                in Cartesian space. If using fractional coordinates set to
                False.
            round_dp (int, optional): The number of decimal places to round the
                miller index to.
            verbose (bool, optional): Whether to print warnings.

        Returns:
            tuple: The Miller index.
        """
    if coords_are_cartesian:
        coords = [self.get_fractional_coords(c) for c in coords]
    coords = np.asarray(coords)
    g = coords.sum(axis=0) / coords.shape[0]
    _, _, vh = np.linalg.svd(coords - g)
    u_norm = vh[2, :]
    return get_integer_index(u_norm, round_dp=round_dp, verbose=verbose)