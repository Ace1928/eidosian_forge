from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_not_equals_non_arraylike(self, index):
    assert not index.equals(list(index))