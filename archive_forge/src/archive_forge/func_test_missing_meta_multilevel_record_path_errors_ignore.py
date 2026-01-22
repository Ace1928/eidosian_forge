import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_missing_meta_multilevel_record_path_errors_ignore(self, missing_metadata):
    result = json_normalize(data=missing_metadata, record_path=['previous_residences', 'cities'], meta='name', errors='ignore')
    ex_data = [['Foo York City', 'Alice'], ['Barmingham', np.nan]]
    columns = ['city_name', 'name']
    expected = DataFrame(ex_data, columns=columns)
    tm.assert_frame_equal(result, expected)