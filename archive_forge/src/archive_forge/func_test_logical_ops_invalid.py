import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_logical_ops_invalid(self, using_infer_string):
    df1 = DataFrame(1.0, index=[1], columns=['A'])
    df2 = DataFrame(True, index=[1], columns=['A'])
    msg = re.escape("unsupported operand type(s) for |: 'float' and 'bool'")
    with pytest.raises(TypeError, match=msg):
        df1 | df2
    df1 = DataFrame('foo', index=[1], columns=['A'])
    df2 = DataFrame(True, index=[1], columns=['A'])
    msg = re.escape("unsupported operand type(s) for |: 'str' and 'bool'")
    if using_infer_string:
        import pyarrow as pa
        with pytest.raises(pa.lib.ArrowNotImplementedError, match='|has no kernel'):
            df1 | df2
    else:
        with pytest.raises(TypeError, match=msg):
            df1 | df2