from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def most_common_values(df):
    return Series({c: s.value_counts().index[0] for c, s in df.items()})