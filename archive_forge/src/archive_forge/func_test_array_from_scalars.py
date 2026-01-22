import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_array_from_scalars(self, data):
    scalars = [data[0], data[1], data[2]]
    result = data._from_sequence(scalars, dtype=data.dtype)
    assert isinstance(result, type(data))