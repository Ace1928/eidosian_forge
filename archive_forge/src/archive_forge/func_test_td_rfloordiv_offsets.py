from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rfloordiv_offsets(self):
    assert offsets.Hour(1) // Timedelta(minutes=25) == 2