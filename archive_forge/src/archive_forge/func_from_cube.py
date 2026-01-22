from __future__ import annotations
import itertools
import json
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from scipy.interpolate import RegularGridInterpolator
from pymatgen.core import Element, Site, Structure
from pymatgen.core.units import ang_to_bohr, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
@classmethod
def from_cube(cls, filename: str | Path) -> Self:
    """
        Initialize the cube object and store the data as data.

        Args:
            filename (str): of the cube to read
        """
    file = zopen(filename, mode='rt')
    file.readline()
    file.readline()
    line = file.readline().split()
    n_atoms = int(line[0])
    line = file.readline().split()
    num_x_voxels = int(line[0])
    voxel_x = np.array([bohr_to_angstrom * float(val) for val in line[1:]])
    line = file.readline().split()
    num_y_voxels = int(line[0])
    voxel_y = np.array([bohr_to_angstrom * float(val) for val in line[1:]])
    line = file.readline().split()
    num_z_voxels = int(line[0])
    voxel_z = np.array([bohr_to_angstrom * float(val) for val in line[1:]])
    sites = []
    for _ in range(n_atoms):
        line = file.readline().split()
        sites.append(Site(line[0], np.multiply(bohr_to_angstrom, list(map(float, line[2:])))))
    structure = Structure(lattice=[voxel_x * num_x_voxels, voxel_y * num_y_voxels, voxel_z * num_z_voxels], species=[s.specie for s in sites], coords=[s.coords for s in sites], coords_are_cartesian=True)
    data = np.reshape(np.array(file.read().split()).astype(float), (num_x_voxels, num_y_voxels, num_z_voxels))
    return cls(structure=structure, data={'total': data})