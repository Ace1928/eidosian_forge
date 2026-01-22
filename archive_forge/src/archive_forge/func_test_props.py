import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
def test_props(forceprop):
    print(forceprop)
    assert forceprop['forces'].shape == (natoms, 3)
    assert forceprop['natoms'] == natoms