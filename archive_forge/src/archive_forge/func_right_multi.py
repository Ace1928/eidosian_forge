import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
@pytest.fixture
def right_multi():
    return DataFrame({'Origin': ['A', 'A', 'B', 'B', 'C', 'C', 'E'], 'Destination': ['A', 'B', 'A', 'B', 'A', 'B', 'F'], 'Period': ['AM', 'AM', 'IP', 'AM', 'OP', 'IP', 'AM'], 'LinkType': ['a', 'b', 'c', 'b', 'a', 'b', 'a'], 'Distance': [100, 80, 90, 80, 75, 35, 55]}, columns=['Origin', 'Destination', 'Period', 'LinkType', 'Distance']).set_index(['Origin', 'Destination', 'Period', 'LinkType'])