from io import StringIO
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas import DataFrame
import pandas._testing as tm

Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
