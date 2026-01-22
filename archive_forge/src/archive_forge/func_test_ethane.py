import pytest
import numpy as np
from ase import io
def test_ethane():
    fname = 'ethane.cml'
    with open(fname, 'w') as fd:
        fd.write(ethane)
    atoms = io.read(fname)
    assert str(atoms.symbols) == 'HCH2CH3'