from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('agg', ['cov', 'corr'])
def test_step_not_implemented_for_cov_corr(agg):
    roll = DataFrame(range(2)).rolling(1, step=2)
    with pytest.raises(NotImplementedError, match='step not implemented'):
        getattr(roll, agg)()