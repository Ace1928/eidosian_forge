import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm

    Return pair or different warnings.

    Useful for testing how several different warnings are handled
    in tm.assert_produces_warning.
    