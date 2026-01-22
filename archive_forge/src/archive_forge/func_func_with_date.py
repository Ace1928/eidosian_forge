from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def func_with_date(batch):
    return Series({'b': datetime(2015, 1, 1), 'c': 2})