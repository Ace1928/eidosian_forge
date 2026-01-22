import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_seaborn():
    seaborn = pytest.importorskip('seaborn')
    tips = DataFrame({'day': pd.date_range('2023', freq='D', periods=5), 'total_bill': range(5)})
    seaborn.stripplot(x='day', y='total_bill', data=tips)