import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
def test_sample_empty(self):
    empty_dc = MinimalDOSCollection([])
    with pytest.raises(IndexError):
        empty_dc._sample(10)
    with pytest.raises(IndexError):
        empty_dc.sample_grid(10)