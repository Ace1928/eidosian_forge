import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.fixture
def missing_metadata():
    return [{'name': 'Alice', 'addresses': [{'number': 9562, 'street': 'Morris St.', 'city': 'Massillon', 'state': 'OH', 'zip': 44646}], 'previous_residences': {'cities': [{'city_name': 'Foo York City'}]}}, {'addresses': [{'number': 8449, 'street': 'Spring St.', 'city': 'Elizabethton', 'state': 'TN', 'zip': 37643}], 'previous_residences': {'cities': [{'city_name': 'Barmingham'}]}}]