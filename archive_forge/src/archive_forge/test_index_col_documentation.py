from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm

Tests that the specified index column (a.k.a "index_col")
is properly handled or inferred during parsing for all of
the parsers defined in parsers.py
