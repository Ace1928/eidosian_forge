import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('orient', ['index', 'columns'])
def test_frame_non_unique_index_raises(self, orient):
    df = DataFrame([['a', 'b'], ['c', 'd']], index=[1, 1], columns=['x', 'y'])
    msg = f"DataFrame index must be unique for orient='{orient}'"
    with pytest.raises(ValueError, match=msg):
        df.to_json(orient=orient)