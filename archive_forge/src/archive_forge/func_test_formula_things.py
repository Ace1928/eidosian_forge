import numpy as np
import pytest
from ase import Atoms
from ase.formula import Formula
def test_formula_things():
    assert Formula('A3B2C2D').format('abc') == 'DB2C2A3'
    assert str(Formula('HHOOO', format='reduce')) == 'H2O3'
    assert Formula('HHOOOUO').format('reduce') == 'H2O3UO'