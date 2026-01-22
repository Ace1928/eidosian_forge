from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_nonstring_object():
    df = pd.DataFrame({'A': ['a', 10, 1.0, ()]})
    col = df.__dataframe__().get_column_by_name('A')
    with pytest.raises(NotImplementedError, match='not supported yet'):
        col.dtype