from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
def test_center_nocell():
    atoms = Atoms('H', positions=[[1.0, 2.0, 3.0]])
    atoms.center()
    assert atoms.positions == pytest.approx(0)