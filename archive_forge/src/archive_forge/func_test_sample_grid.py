from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
def test_sample_grid(self, sparse_dos):
    min_dos = sparse_dos.sample_grid(10, xmax=5, padding=3, width=0.1)
    assert min_dos.get_energies()[0] == pytest.approx(1.2 - 3 * 0.1)
    max_dos = sparse_dos.sample_grid(10, xmin=0, padding=2, width=0.2)
    assert max_dos.get_energies()[-1] == pytest.approx(5 + 2 * 0.2)
    default_dos = sparse_dos.sample_grid(10)
    assert np.allclose(default_dos.get_energies(), np.linspace(0.9, 5.3, 10))
    dos0 = sparse_dos._sample(np.linspace(0.9, 5.3, 10))
    assert np.allclose(default_dos.get_weights(), dos0)