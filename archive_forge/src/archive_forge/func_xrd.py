from pathlib import Path
import numpy as np
import pytest
from ase.utils.xrdebye import XrDebye, wavelengths
from ase.cluster.cubic import FaceCenteredCubic
@pytest.fixture
def xrd():
    atoms = FaceCenteredCubic('Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [6, 8, 8], 4.09)
    return XrDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.04, method='Iwasa', alpha=1.01, warn=True)