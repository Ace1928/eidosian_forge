from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def check_skip(data, op_name):
    if isinstance(data.dtype, pd.BooleanDtype) and 'sub' in op_name:
        pytest.skip('subtract not implemented for boolean')