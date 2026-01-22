from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def using_pyarrow(dtype):
    return dtype in ('string[pyarrow]', 'string[pyarrow_numpy]')