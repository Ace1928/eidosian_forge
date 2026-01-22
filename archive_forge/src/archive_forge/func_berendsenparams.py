import pytest
from ase import Atoms
from ase.units import fs, GPa, bar
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
@pytest.fixture(scope='module')
def berendsenparams():
    """Parameters for the two Berendsen algorithms."""
    Bgold = 220.0 * GPa
    nvtparam = dict(temperature_K=300, taut=1000 * fs)
    nptparam = dict(temperature_K=300, pressure_au=5000 * bar, taut=1000 * fs, taup=1000 * fs, compressibility_au=1 / Bgold)
    return dict(nvt=nvtparam, npt=nptparam)