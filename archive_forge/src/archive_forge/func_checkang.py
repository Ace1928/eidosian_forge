from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
def checkang(a, b, phi):
    """Check the angle between two vectors."""
    cosphi = np.dot(a, b) / sqrt(np.dot(a, a) * np.dot(b, b))
    assert np.abs(cosphi - cos(phi)) < 1e-10