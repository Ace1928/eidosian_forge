from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.skipif(PD_LT_1_0_0, reason='bug in old pandas')
def test_index_like():
    idx = np.empty((100, 2))
    with pytest.raises(TypeError, match='index must be a pandas'):
        DeterministicTerm._index_like(idx)