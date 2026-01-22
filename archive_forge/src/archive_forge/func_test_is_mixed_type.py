from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_is_mixed_type(self, float_frame, float_string_frame):
    assert not float_frame._is_mixed_type
    assert float_string_frame._is_mixed_type