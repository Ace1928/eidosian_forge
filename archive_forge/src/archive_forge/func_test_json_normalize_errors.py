import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_json_normalize_errors(self, missing_metadata):
    msg = "Key 'name' not found. To replace missing values of 'name' with np.nan, pass in errors='ignore'"
    with pytest.raises(KeyError, match=msg):
        json_normalize(data=missing_metadata, record_path='addresses', meta='name', errors='raise')