from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_h2_ls(cp2k_factory, atoms):
    inp = '&FORCE_EVAL\n               &DFT\n                 &QS\n                   LS_SCF ON\n                 &END QS\n               &END DFT\n             &END FORCE_EVAL'
    calc = cp2k_factory.calc(label='test_H2_LS', inp=inp)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -30.6989581747
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 5e-07