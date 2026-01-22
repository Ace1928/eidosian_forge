import pytest
import pyarrow as pa
from pyarrow import Codec
from pyarrow import fs
import numpy as np
@pytest.fixture(scope='session')
def varargs_agg_func_fixture():
    """
    Register a unary aggregate function
    """
    from pyarrow import compute as pc

    def func(ctx, *args):
        sum = 0.0
        for arg in args:
            sum += np.nanmean(arg)
        return pa.scalar(sum)
    func_name = 'sum_mean'
    func_doc = {'summary': 'Varargs aggregate', 'description': 'Varargs aggregate'}
    pc.register_aggregate_function(func, func_name, func_doc, {'x': pa.int64(), 'y': pa.float64()}, pa.float64())
    return (func, func_name)