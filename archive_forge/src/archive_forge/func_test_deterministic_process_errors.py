from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_deterministic_process_errors(time_index):
    with pytest.raises(ValueError, match='seasonal and fourier'):
        DeterministicProcess(time_index, seasonal=True, fourier=2, period=5)
    with pytest.raises(TypeError, match='All additional terms'):
        DeterministicProcess(time_index, seasonal=True, additional_terms=[1])