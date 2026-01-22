import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
@pytest.fixture
def deep_nested():
    return [{'country': 'USA', 'states': [{'name': 'California', 'cities': [{'name': 'San Francisco', 'pop': 12345}, {'name': 'Los Angeles', 'pop': 12346}]}, {'name': 'Ohio', 'cities': [{'name': 'Columbus', 'pop': 1234}, {'name': 'Cleveland', 'pop': 1236}]}]}, {'country': 'Germany', 'states': [{'name': 'Bayern', 'cities': [{'name': 'Munich', 'pop': 12347}]}, {'name': 'Nordrhein-Westfalen', 'cities': [{'name': 'Duesseldorf', 'pop': 1238}, {'name': 'Koeln', 'pop': 1239}]}]}]