from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_h2_pbe(cp2k_factory, atoms):
    calc = cp2k_factory.calc(xc='PBE', label='test_H2_PBE')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -31.5917284949
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10