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
def find_neighbors(label: np.ndarray, nx: int, ny: int, nz: int) -> list[np.ndarray]:
    """Given a cube index, find the neighbor cube indices.

    Args:
        label: (array) (n,) or (n x 3) indice array
        nx: (int) number of cells in y direction
        ny: (int) number of cells in y direction
        nz: (int) number of cells in z direction

    Returns:
        Neighbor cell indices.
    """
    array = [[-1, 0, 1]] * 3
    neighbor_vectors = np.array(list(itertools.product(*array)), dtype=int)
    label3d = _one_to_three(label, ny, nz) if np.shape(label)[1] == 1 else label
    all_labels = label3d[:, None, :] - neighbor_vectors[None, :, :]
    filtered_labels = []
    for labels in all_labels:
        ind = (labels[:, 0] < nx) * (labels[:, 1] < ny) * (labels[:, 2] < nz) * np.all(labels > -1e-05, axis=1)
        filtered_labels.append(labels[ind])
    return filtered_labels