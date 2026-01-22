import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
def set_timestep(self, timestep):
    self.dt = timestep