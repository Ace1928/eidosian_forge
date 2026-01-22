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
@ase.utils.deprecated('Please use phonons.compute_lattice_vectors() instead of .lattice_vectors()')
def lattice_vectors(self):
    return self.compute_lattice_vectors()