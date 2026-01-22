import pytest
from ase import Atoms
from ase.units import fs, GPa, bar
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
Make an atomic system with equilibrated temperature and pressure.