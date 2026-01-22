from __future__ import annotations
import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.constants import physical_constants, speed_of_light
from scipy.integrate import simps
from scipy.interpolate import interp1d
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.due import Doi, due
def parse_dielectric_data(data):
    """
    Convert a set of 2D vasprun formatted dielectric data to
    the eigenvalues of each corresponding 3x3 symmetric numpy matrices.

    Args:
        data (list): length N list of dielectric data. Each entry should be
            a list of ``[xx, yy, zz, xy , xz, yz ]`` dielectric tensor elements.

    Returns:
        np.array: a Nx3 numpy array. Each row contains the eigenvalues
            for the corresponding row in `data`.
    """
    return np.array([np.linalg.eig(to_matrix(*eps))[0] for eps in data])