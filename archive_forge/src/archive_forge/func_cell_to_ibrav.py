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
def cell_to_ibrav(cell, ibrav):
    """
    Calculate the appropriate `celldm(..)` parameters for the given ibrav
    using the given cell. The units for `celldm(..)` are Bohr.

    Does minimal checking of the cell shape, so it is possible to create
    a nonsense structure if the ibrav is inapproprite for the cell. These
    are derived to be symmetric with the routine for constructing the cell
    from ibrav parameters so directions of some vectors may be unexpected.

    Parameters
    ----------
    cell : np.array
        A 3x3 representation of a unit cell
    ibrav : int
        Bravais-lattice index according to the pw.x designations.

    Returns
    -------
    parameters : dict
        A dictionary with all the necessary `celldm(..)` keys assigned
        necessary values (in units of Bohr). Also includes `ibrav` so it
        can be passed back to `ibrav_to_cell`.

    Raises
    ------
    NotImplementedError
        Only a limited number of ibrav settings can be parsed. An error
        is raised if the ibrav interpretation is not implemented.
    """
    parameters = {'ibrav': ibrav}
    if ibrav == 1:
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
    elif ibrav in [2, 3, -3]:
        parameters['celldm(1)'] = cell[0][2] * 2 / units['Bohr']
    elif ibrav in [4, 6]:
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav in [5, -5]:
        a = np.linalg.norm(cell[0])
        cosab = np.dot(cell[0], cell[1]) / a ** 2
        parameters['celldm(1)'] = a / units['Bohr']
        parameters['celldm(4)'] = cosab
    elif ibrav == 7:
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav == 8:
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav in [9, -9]:
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] * 2 / cell[0][0]
    elif ibrav in [10, 11]:
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
    elif ibrav == 12:
        b = (cell[1][0] ** 2 + cell[1][1] ** 2) ** 0.5
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(2)'] = b / cell[0][0]
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
        parameters['celldm(4)'] = cell[1][0] / b
    elif ibrav == -12:
        c = (cell[2][0] ** 2 + cell[2][2] ** 2) ** 0.5
        parameters['celldm(1)'] = cell[0][0] / units['Bohr']
        parameters['celldm(2)'] = cell[1][1] / cell[0][0]
        parameters['celldm(3)'] = c / cell[0][0]
        parameters['celldm(4)'] = cell[2][0] / c
    elif ibrav == 13:
        b = (cell[1][0] ** 2 + cell[1][1] ** 2) ** 0.5
        parameters['celldm(1)'] = cell[0][0] * 2 / units['Bohr']
        parameters['celldm(2)'] = b / (cell[0][0] * 2)
        parameters['celldm(3)'] = cell[2][2] / cell[0][0]
        parameters['celldm(4)'] = cell[1][0] / b
    elif ibrav == 14:
        a, b, c = np.linalg.norm(cell, axis=1)
        cosbc = np.dot(cell[1], cell[2]) / (b * c)
        cosac = np.dot(cell[0], cell[2]) / (a * c)
        cosab = np.dot(cell[0], cell[1]) / (a * b)
        parameters['celldm(1)'] = a / units['Bohr']
        parameters['celldm(2)'] = b / a
        parameters['celldm(3)'] = c / a
        parameters['celldm(4)'] = cosbc
        parameters['celldm(5)'] = cosac
        parameters['celldm(6)'] = cosab
    else:
        raise NotImplementedError('ibrav = {0} is not implemented'.format(ibrav))
    return parameters