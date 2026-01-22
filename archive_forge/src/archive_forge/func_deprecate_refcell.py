from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
@staticmethod
def deprecate_refcell(kwargs: dict):
    if 'refcell' in kwargs:
        warnings.warn('Keyword refcell of Phonons is deprecated.Please use center_refcell (bool)', FutureWarning)
        kwargs['center_refcell'] = bool(kwargs['refcell'])
        kwargs.pop('refcell')
    return kwargs