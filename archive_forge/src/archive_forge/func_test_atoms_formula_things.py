import numpy as np
import pytest
from ase import Atoms
from ase.formula import Formula
def test_atoms_formula_things():
    assert Atoms('MoS2').get_chemical_formula() == 'MoS2'
    assert Atoms('SnO2').get_chemical_formula(mode='metal') == 'SnO2'