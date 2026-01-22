import pytest
import pandas._testing as tm
from pandas.io.parsers import read_csv
@pytest.fixture(params=[True, False])
def merge_cells(request):
    return request.param