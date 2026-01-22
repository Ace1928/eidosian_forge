import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def test_system_changes():
    for atoms in systems_minimum():
        atoms.calc.calculate(atoms, system_changes=['positions'])
        assert atoms.get_potential_energy() == reference_potential_energy