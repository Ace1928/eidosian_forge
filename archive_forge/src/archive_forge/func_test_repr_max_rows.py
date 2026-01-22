from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_max_rows(self):
    with option_context('display.max_rows', None):
        str(Series(range(1001)))