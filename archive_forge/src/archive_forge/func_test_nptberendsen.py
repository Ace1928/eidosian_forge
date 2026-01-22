import pytest
from ase import Atoms
from ase.units import fs, GPa, bar
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
@pytest.mark.slow
def test_nptberendsen(asap3, equilibrated, berendsenparams, allraise):
    t, p = propagate(Atoms(equilibrated), asap3, NPTBerendsen, berendsenparams['npt'])
    assert abs(t - berendsenparams['npt']['temperature_K']) < 1.0
    assert abs(p - berendsenparams['npt']['pressure_au']) < 25.0 * bar