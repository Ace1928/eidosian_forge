from ase.units import fs
from ase.build import bulk
from ase.md import Langevin
from ase.md.fix import FixRotation
from ase.utils import seterr
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
import numpy as np
def test_fixrotation_asap(asap3):
    rng = np.random.RandomState(123)
    with seterr(all='raise'):
        atoms = bulk('Au', cubic=True).repeat((3, 3, 10))
        atoms.pbc = False
        atoms.center(vacuum=5.0 + np.max(atoms.cell) / 2)
        print(atoms)
        atoms.calc = asap3.EMT()
        MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True, rng=rng)
        Stationary(atoms)
        check_inertia(atoms)
        with Langevin(atoms, timestep=20 * fs, temperature_K=300, friction=0.001, logfile='-', loginterval=500, rng=rng) as md:
            fx = FixRotation(atoms)
            md.attach(fx)
            md.run(steps=1000)
        check_inertia(atoms)