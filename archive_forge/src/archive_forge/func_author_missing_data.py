import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.fixture
def author_missing_data():
    return [{'info': None}, {'info': {'created_at': '11/08/1993', 'last_updated': '26/05/2012'}, 'author_name': {'first': 'Jane', 'last_name': 'Doe'}}]