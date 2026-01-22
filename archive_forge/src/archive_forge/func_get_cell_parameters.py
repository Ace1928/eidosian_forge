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
def get_cell_parameters(lines, alat=None):
    """Parse unit cell from CELL_PARAMETERS card.

    Parameters
    ----------
    lines : list[str]
        A list with lines containing the CELL_PARAMETERS card.
    alat : float | None
        Unit of lattice vectors in Angstrom. Only used if the card is
        given in units of alat. alat must be None if CELL_PARAMETERS card
        is in Bohr or Angstrom. For output files, alat will be parsed from
        the card header and used in preference to this value.

    Returns
    -------
    cell : np.array | None
        Cell parameters as a 3x3 array in Angstrom. If no cell is found
        None will be returned instead.
    cell_alat : float | None
        If a value for alat is given in the card header, this is also
        returned, otherwise this will be None.

    Raises
    ------
    ValueError
        If CELL_PARAMETERS are given in units of bohr or angstrom
        and alat is not
    """
    cell = None
    cell_alat = None
    trimmed_lines = (line for line in lines if line.strip() and (not line[0] == '#'))
    for line in trimmed_lines:
        if line.strip().startswith('CELL_PARAMETERS'):
            if cell is not None:
                raise ValueError('CELL_PARAMETERS specified multiple times')
            if 'bohr' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in &SYSTEM celldm/A and CELL_PARAMETERS bohr')
                cell_units = units['Bohr']
            elif 'angstrom' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in &SYSTEM celldm/A and CELL_PARAMETERS angstrom')
                cell_units = 1.0
            elif 'alat' in line.lower():
                if '=' in line:
                    alat = float(line.strip(') \n').split()[-1]) * units['Bohr']
                    cell_alat = alat
                elif alat is None:
                    raise ValueError('Lattice parameters must be set in &SYSTEM for alat units')
                cell_units = alat
            elif alat is None:
                cell_units = units['Bohr']
            else:
                cell_units = alat
            cell = [[ffloat(x) for x in next(trimmed_lines).split()[:3]], [ffloat(x) for x in next(trimmed_lines).split()[:3]], [ffloat(x) for x in next(trimmed_lines).split()[:3]]]
            cell = np.array(cell) * cell_units
    return (cell, cell_alat)