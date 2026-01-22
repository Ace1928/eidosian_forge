from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_elems_unknown_column(parser, geom_df):
    with pytest.raises(KeyError, match='no valid column'):
        geom_df.to_xml(elem_cols=['shape', 'degree', 'sides'], parser=parser)