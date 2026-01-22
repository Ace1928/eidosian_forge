from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_complex_sorting(self):
    x17 = np.array([complex(i) for i in range(17)], dtype=object)
    msg = "'[<>]' not supported between instances of .*"
    with pytest.raises(TypeError, match=msg):
        algos.factorize(x17[::-1], sort=True)