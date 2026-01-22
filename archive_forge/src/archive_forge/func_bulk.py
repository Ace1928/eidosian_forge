from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@functools.wraps(newbulk)
def bulk(*args, **kwargs):
    warnings.warn('Use ase.build.bulk() instead', stacklevel=2)
    return newbulk(*args, **kwargs)