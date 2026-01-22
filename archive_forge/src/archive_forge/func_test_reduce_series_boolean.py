from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
@pytest.mark.parametrize('skipna', [True, False])
def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
    op_name = all_boolean_reductions
    ser = pd.Series(data)
    if not self._supports_reduction(ser, op_name):
        msg = '[Cc]annot perform|Categorical is not ordered for operation|does not support reduction|'
        with pytest.raises(TypeError, match=msg):
            getattr(ser, op_name)(skipna=skipna)
    else:
        self.check_reduce(ser, op_name, skipna)