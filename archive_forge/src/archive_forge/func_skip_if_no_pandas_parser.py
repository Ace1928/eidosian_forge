import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def skip_if_no_pandas_parser(parser):
    if parser != 'pandas':
        pytest.skip(f'cannot evaluate with parser={parser}')