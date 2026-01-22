from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
def calculate_and_save_static_polarizability(self, atoms):
    exobj = self.vib._new_exobj()
    excitation_data = exobj.calculate(atoms)
    np.savetxt(self._exname, excitation_data)