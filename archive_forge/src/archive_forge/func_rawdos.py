import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.fixture
def rawdos(self):
    return RawDOSData([1.0, 2.0, 4.0], [2.0, 3.0, 2.0], info={'my_key': 'my_value'})