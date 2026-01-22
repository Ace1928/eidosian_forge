import pytest
from datashader.datashape.user import issubschema, validate
from datashader.datashape import dshape
from datetime import date, time, datetime
import numpy as np
def test_numeric_tower():
    assert validate(np.integer, np.int32(1))
    assert validate(np.number, np.int32(1))