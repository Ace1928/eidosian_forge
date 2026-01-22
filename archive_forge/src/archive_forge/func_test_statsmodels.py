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
def test_statsmodels():
    smf = pytest.importorskip('statsmodels.formula.api')
    df = DataFrame({'Lottery': range(5), 'Literacy': range(5), 'Pop1831': range(100, 105)})
    smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=df).fit()