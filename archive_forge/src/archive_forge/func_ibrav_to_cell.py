import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
def ibrav_to_cell(system):
    """
    Convert a value of ibrav to a cell. Any unspecified lattice dimension
    is set to 0.0, but will not necessarily raise an error. Also return the
    lattice parameter.

    Parameters
    ----------
    system : dict
        The &SYSTEM section of the input file, containing the 'ibrav' setting,
        and either celldm(1)..(6) or a, b, c, cosAB, cosAC, cosBC.

    Returns
    -------
    alat, cell : float, np.array
        Cell parameter in Angstrom, and
        The 3x3 array representation of the cell.

    Raises
    ------
    KeyError
        Raise an error if any required keys are missing.
    NotImplementedError
        Only a limited number of ibrav settings can be parsed. An error
        is raised if the ibrav interpretation is not implemented.
    """
    if 'celldm(1)' in system and 'a' in system:
        raise KeyError('do not specify both celldm and a,b,c!')
    elif 'celldm(1)' in system:
        alat = system['celldm(1)'] * units['Bohr']
        b_over_a = system.get('celldm(2)', 0.0)
        c_over_a = system.get('celldm(3)', 0.0)
        cosab = system.get('celldm(4)', 0.0)
        cosac = system.get('celldm(5)', 0.0)
        cosbc = 0.0
        if system['ibrav'] == 14:
            cosbc = system.get('celldm(4)', 0.0)
            cosac = system.get('celldm(5)', 0.0)
            cosab = system.get('celldm(6)', 0.0)
    elif 'a' in system:
        alat = system['a']
        b_over_a = system.get('b', 0.0) / alat
        c_over_a = system.get('c', 0.0) / alat
        cosab = system.get('cosab', 0.0)
        cosac = system.get('cosac', 0.0)
        cosbc = system.get('cosbc', 0.0)
    else:
        raise KeyError('Missing celldm(1) or a cell parameter.')
    if system['ibrav'] == 1:
        cell = np.identity(3) * alat
    elif system['ibrav'] == 2:
        cell = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]]) * (alat / 2)
    elif system['ibrav'] == 3:
        cell = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]]) * (alat / 2)
    elif system['ibrav'] == -3:
        cell = np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]) * (alat / 2)
    elif system['ibrav'] == 4:
        cell = np.array([[1.0, 0.0, 0.0], [-0.5, 0.5 * 3 ** 0.5, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 5:
        tx = ((1.0 - cosab) / 2.0) ** 0.5
        ty = ((1.0 - cosab) / 6.0) ** 0.5
        tz = ((1 + 2 * cosab) / 3.0) ** 0.5
        cell = np.array([[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]]) * alat
    elif system['ibrav'] == -5:
        ty = ((1.0 - cosab) / 6.0) ** 0.5
        tz = ((1 + 2 * cosab) / 3.0) ** 0.5
        a_prime = alat / 3 ** 0.5
        u = tz - 2 * 2 ** 0.5 * ty
        v = tz + 2 ** 0.5 * ty
        cell = np.array([[u, v, v], [v, u, v], [v, v, u]]) * a_prime
    elif system['ibrav'] == 6:
        cell = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 7:
        cell = np.array([[1.0, -1.0, c_over_a], [1.0, 1.0, c_over_a], [-1.0, -1.0, c_over_a]]) * (alat / 2)
    elif system['ibrav'] == 8:
        cell = np.array([[1.0, 0.0, 0.0], [0.0, b_over_a, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 9:
        cell = np.array([[1.0 / 2.0, b_over_a / 2.0, 0.0], [-1.0 / 2.0, b_over_a / 2.0, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == -9:
        cell = np.array([[1.0 / 2.0, -b_over_a / 2.0, 0.0], [1.0 / 2.0, b_over_a / 2.0, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 10:
        cell = np.array([[1.0 / 2.0, 0.0, c_over_a / 2.0], [1.0 / 2.0, b_over_a / 2.0, 0.0], [0.0, b_over_a / 2.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 11:
        cell = np.array([[1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0], [-1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0], [-1.0, 2.0, -b_over_a / 2.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 12:
        sinab = (1.0 - cosab ** 2) ** 0.5
        cell = np.array([[1.0, 0.0, 0.0], [b_over_a * cosab, b_over_a * sinab, 0.0], [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == -12:
        sinac = (1.0 - cosac ** 2) ** 0.5
        cell = np.array([[1.0, 0.0, 0.0], [0.0, b_over_a, 0.0], [c_over_a * cosac, 0.0, c_over_a * sinac]]) * alat
    elif system['ibrav'] == 13:
        sinab = (1.0 - cosab ** 2) ** 0.5
        cell = np.array([[1.0 / 2.0, 0.0, -c_over_a / 2.0], [b_over_a * cosab, b_over_a * sinab, 0.0], [1.0 / 2.0, 0.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 14:
        sinab = (1.0 - cosab ** 2) ** 0.5
        v3 = [c_over_a * cosac, c_over_a * (cosbc - cosac * cosab) / sinab, c_over_a * (1 + 2 * cosbc * cosac * cosab - cosbc ** 2 - cosac ** 2 - cosab ** 2) ** 0.5 / sinab]
        cell = np.array([[1.0, 0.0, 0.0], [b_over_a * cosab, b_over_a * sinab, 0.0], v3]) * alat
    else:
        raise NotImplementedError('ibrav = {0} is not implemented'.format(system['ibrav']))
    return (alat, cell)