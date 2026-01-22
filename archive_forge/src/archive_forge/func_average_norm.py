import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def average_norm(self, i, j, dx):
    """Average norm between images i and j

        Args:
            i (int): left image
            j (int): right image
            dx (array): vector

        Returns:
            norm: norm of vector wrt average of precons at i and j
        """
    return np.sqrt(0.5 * (self.precon[i].dot(dx, dx) + self.precon[j].dot(dx, dx)))