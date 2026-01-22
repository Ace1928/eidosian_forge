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
def write_mode(self, n=None, kT=units.kB * 300, nimages=30):
    """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
    if n is None:
        for index, energy in enumerate(self.get_energies()):
            if abs(energy) > 1e-05:
                self.write_mode(n=index, kT=kT, nimages=nimages)
        return
    else:
        n %= len(self.get_energies())
    with ase.io.Trajectory('%s.%d.traj' % (self.name, n), 'w') as traj:
        for image in self.get_vibrations().iter_animated_mode(n, temperature=kT, frames=nimages):
            traj.write(image)