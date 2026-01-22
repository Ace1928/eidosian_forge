import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm

Tests that can be parametrized over _any_ Index object.
