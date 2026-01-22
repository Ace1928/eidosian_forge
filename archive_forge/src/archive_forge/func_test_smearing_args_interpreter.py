from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.mark.parametrize('inputs, expected', smearing_args)
def test_smearing_args_interpreter(self, inputs, expected):
    assert GridDOSData._interpret_smearing_args(**inputs) == expected