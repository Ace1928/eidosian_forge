import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def check_row_subclass(row):
    assert isinstance(row, tm.SubclassedSeries)