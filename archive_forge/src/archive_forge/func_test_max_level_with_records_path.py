import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.mark.parametrize('max_level,expected', [(0, [{'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}, 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}, {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}, 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}]), (1, [{'TextField': 'Some text', 'UserField.Id': 'ID001', 'UserField.Name': 'Name001', 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}, {'TextField': 'Some text', 'UserField.Id': 'ID001', 'UserField.Name': 'Name001', 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}])])
def test_max_level_with_records_path(self, max_level, expected):
    test_input = [{'CreatedBy': {'Name': 'User001'}, 'Lookup': [{'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}, {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}], 'Image': {'a': 'b'}, 'tags': [{'foo': 'something', 'bar': 'else'}, {'foo': 'something2', 'bar': 'else2'}]}]
    result = json_normalize(test_input, record_path=['Lookup'], meta=[['CreatedBy'], ['Image']], max_level=max_level)
    expected_df = DataFrame(data=expected, columns=result.columns.values)
    tm.assert_equal(expected_df, result)