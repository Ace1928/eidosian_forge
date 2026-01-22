import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
def test_str_uses_object():
    result = SparseDtype(str).subtype
    assert result == np.dtype('object')