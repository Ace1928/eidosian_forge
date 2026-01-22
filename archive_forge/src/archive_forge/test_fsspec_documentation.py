import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
Regression test for writing to a not-yet-existent GCS Parquet file.