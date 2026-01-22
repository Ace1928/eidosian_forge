from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_error_len_mismatch(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    other = [scalar] * (len(data) - 1)
    err = ValueError
    msg = '|'.join(['operands could not be broadcast together with shapes \\(3,\\) \\(4,\\)', 'operands could not be broadcast together with shapes \\(4,\\) \\(3,\\)'])
    if data.dtype.kind == 'b' and all_arithmetic_operators.strip('_') in ['sub', 'rsub']:
        err = TypeError
        msg = 'numpy boolean subtract, the `\\-` operator, is not supported, use the bitwise_xor, the `\\^` operator, or the logical_xor function instead'
    elif is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        err = NotImplementedError
    for other in [other, np.array(other)]:
        with pytest.raises(err, match=msg):
            op(data, other)
        s = pd.Series(data)
        with pytest.raises(err, match=msg):
            op(s, other)