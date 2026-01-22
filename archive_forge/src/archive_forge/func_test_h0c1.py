import numpy as np
import pytest
from ase import Atoms
from ase.formula import Formula
def test_h0c1():
    f = Formula.from_dict({'H': 0, 'C': 1})
    assert f.format('hill') == 'C'
    with pytest.raises(ValueError):
        Formula.from_dict({'H': -1})
    with pytest.raises(ValueError):
        Formula.from_dict({'H': 1.5})
    with pytest.raises(ValueError):
        Formula.from_dict({7: 1})