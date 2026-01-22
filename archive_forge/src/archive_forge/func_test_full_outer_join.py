import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_full_outer_join(self, df, df2):
    joined_key2 = merge(df, df2, on='key2', how='outer')
    _check_join(df, df2, joined_key2, ['key2'], how='outer')
    joined_both = merge(df, df2, how='outer')
    _check_join(df, df2, joined_both, ['key1', 'key2'], how='outer')