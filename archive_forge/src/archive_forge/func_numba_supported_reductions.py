import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture(params=[('mean', {}), ('var', {'ddof': 1}), ('var', {'ddof': 0}), ('std', {'ddof': 1}), ('std', {'ddof': 0}), ('sum', {}), ('min', {}), ('max', {}), ('sum', {'min_count': 2}), ('min', {'min_count': 2}), ('max', {'min_count': 2})], ids=['mean', 'var_1', 'var_0', 'std_1', 'std_0', 'sum', 'min', 'max', 'sum-min_count', 'min-min_count', 'max-min_count'])
def numba_supported_reductions(request):
    """reductions supported with engine='numba'"""
    return request.param