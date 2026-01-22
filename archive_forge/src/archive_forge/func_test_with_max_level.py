import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.mark.parametrize('max_level, expected', [(None, [{'CreatedBy.Name': 'User001', 'Lookup.TextField': 'Some text', 'Lookup.UserField.Id': 'ID001', 'Lookup.UserField.Name': 'Name001', 'Image.a': 'b'}]), (0, [{'CreatedBy': {'Name': 'User001'}, 'Lookup': {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}, 'Image': {'a': 'b'}}]), (1, [{'CreatedBy.Name': 'User001', 'Lookup.TextField': 'Some text', 'Lookup.UserField': {'Id': 'ID001', 'Name': 'Name001'}, 'Image.a': 'b'}])])
def test_with_max_level(self, max_level, expected, max_level_test_input_data):
    output = nested_to_record(max_level_test_input_data, max_level=max_level)
    assert output == expected