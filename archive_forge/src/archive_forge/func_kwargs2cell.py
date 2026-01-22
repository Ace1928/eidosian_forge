import os
import re
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Hartree
from ase.utils import reader
def kwargs2cell(kwargs):
    kwargs = normalize_keywords(kwargs)
    if boxshape_is_ase_compatible(kwargs):
        kwargs.pop('boxshape', None)
        if 'lsize' in kwargs:
            Lsize = kwargs.pop('lsize')
            if not isinstance(Lsize, list):
                Lsize = [[Lsize] * 3]
            assert len(Lsize) == 1
            cell = np.array([2 * float(l) for l in Lsize[0]])
        elif 'latticeparameters' in kwargs:
            latparam = np.array(kwargs.pop('latticeparameters'), float).T
            cell = np.array(kwargs.pop('latticevectors', np.eye(3)), float)
            for a, vec in zip(latparam, cell):
                vec *= a
            assert cell.shape == (3, 3)
    else:
        cell = None
    return (cell, kwargs)