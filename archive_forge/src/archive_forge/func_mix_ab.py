from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def mix_ab() -> dict[str, list[int | str]]:
    return {'a': list(range(4)), 'b': list('ab..')}