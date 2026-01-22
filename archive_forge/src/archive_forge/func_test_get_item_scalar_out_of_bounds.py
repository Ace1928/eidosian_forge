from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
@pytest.mark.parametrize('index', [-1000, -6, 5, 1000])
def test_get_item_scalar_out_of_bounds(index):
    rarray = RaggedArray([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]])
    with pytest.raises(IndexError):
        rarray[index]