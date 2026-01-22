import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture
def custom_dialect():
    dialect_name = 'weird'
    dialect_kwargs = {'doublequote': False, 'escapechar': '~', 'delimiter': ':', 'skipinitialspace': False, 'quotechar': '~', 'quoting': 3}
    return (dialect_name, dialect_kwargs)