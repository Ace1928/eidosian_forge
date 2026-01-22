from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def noddy(value, weight):
    out = np.array(value * weight).repeat(3)
    return out