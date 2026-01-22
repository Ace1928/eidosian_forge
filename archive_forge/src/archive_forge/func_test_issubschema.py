import pytest
from datashader.datashape.user import issubschema, validate
from datashader.datashape import dshape
from datetime import date, time, datetime
import numpy as np
def test_issubschema():
    assert issubschema('int', 'int')
    assert not issubschema('int', 'float32')
    assert issubschema('2 * int', '2 * int')
    assert not issubschema('2 * int', '3 * int')