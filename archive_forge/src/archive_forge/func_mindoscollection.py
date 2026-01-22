import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.fixture
def mindoscollection(self, rawdos, another_rawdos):
    return MinimalDOSCollection([rawdos, another_rawdos])