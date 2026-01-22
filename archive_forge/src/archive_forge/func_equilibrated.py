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
def equilibrated(asap3, berendsenparams):
    """Make an atomic system with equilibrated temperature and pressure."""
    rng = np.random.RandomState(42)
    atoms = bulk('Au', cubic=True).repeat((3, 3, 3))
    atoms.calc = asap3.EMT()
    MaxwellBoltzmannDistribution(atoms, temperature_K=100, force_temp=True, rng=rng)
    Stationary(atoms)
    assert abs(atoms.get_temperature() - 100) < 0.0001
    with NPTBerendsen(atoms, timestep=20 * fs, logfile='-', loginterval=200, **berendsenparams['npt']) as md:
        md.run(steps=1000)
    T = atoms.get_temperature()
    pres = -atoms.get_stress(include_ideal_gas=True)[:3].sum() / 3 / GPa * 10000
    print('Temperature: {:.2f} K    Pressure: {:.2f} bar'.format(T, pres))
    return atoms