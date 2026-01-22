from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
def test_center_empty():
    atoms = Atoms()
    atoms.center()
    assert atoms == Atoms()