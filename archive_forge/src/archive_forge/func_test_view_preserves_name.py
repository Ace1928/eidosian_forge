import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_view_preserves_name(index):
    assert index.view().name == index.name