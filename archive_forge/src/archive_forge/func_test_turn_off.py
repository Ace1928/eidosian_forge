import unittest.mock as mock
import pytest
import modin.pandas as pd
from modin.config import Engine
@pytest.mark.parametrize('set_benchmark_mode', [False], indirect=True)
def test_turn_off(set_benchmark_mode):
    df = pd.DataFrame([0])
    with mock.patch(wait_method) as wait:
        df.dropna()
    wait.assert_not_called()