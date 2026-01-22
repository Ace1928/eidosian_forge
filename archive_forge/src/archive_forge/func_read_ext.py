import pytest
import pandas._testing as tm
from pandas.io.parsers import read_csv
@pytest.fixture(params=['.xls', '.xlsx', '.xlsm', '.ods', '.xlsb'])
def read_ext(request):
    """
    Valid extensions for reading Excel files.
    """
    return request.param