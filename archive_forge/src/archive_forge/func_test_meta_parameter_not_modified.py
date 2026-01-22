import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_meta_parameter_not_modified(self):
    data = [{'foo': 'hello', 'bar': 'there', 'data': [{'foo': 'something', 'bar': 'else'}, {'foo': 'something2', 'bar': 'else2'}]}]
    COLUMNS = ['foo', 'bar']
    result = json_normalize(data, 'data', meta=COLUMNS, meta_prefix='meta')
    assert COLUMNS == ['foo', 'bar']
    for val in ['metafoo', 'metabar', 'foo', 'bar']:
        assert val in result