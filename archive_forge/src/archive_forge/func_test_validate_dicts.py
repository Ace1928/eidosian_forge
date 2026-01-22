import pytest
from datashader.datashape.user import issubschema, validate
from datashader.datashape import dshape
from datetime import date, time, datetime
import numpy as np
@min_np
def test_validate_dicts():
    assert validate('{x: int, y: int}', {'x': 1, 'y': 2})
    assert not validate('{x: int, y: int}', {'x': 1, 'y': 2.0})
    assert not validate('{x: int, y: int}', {'x': 1, 'z': 2})
    assert validate('var * {x: int, y: int}', [{'x': 1, 'y': 2}])
    assert validate('var * {x: int, y: int}', [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}])