import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
from pandas._config import using_copy_on_write
from pandas._config.config import _get_option
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
@pytest.fixture(params=[datetime.datetime.now(datetime.timezone.utc), datetime.datetime.now(datetime.timezone.min), datetime.datetime.now(datetime.timezone.max), datetime.datetime.strptime('2019-01-04T16:41:24+0200', '%Y-%m-%dT%H:%M:%S%z'), datetime.datetime.strptime('2019-01-04T16:41:24+0215', '%Y-%m-%dT%H:%M:%S%z'), datetime.datetime.strptime('2019-01-04T16:41:24-0200', '%Y-%m-%dT%H:%M:%S%z'), datetime.datetime.strptime('2019-01-04T16:41:24-0215', '%Y-%m-%dT%H:%M:%S%z')])
def timezone_aware_date_list(request):
    return request.param