from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_list_raises(string_series):
    with pytest.raises(TypeError, match="'list' object is not callable"):
        string_series.map([lambda x: x])