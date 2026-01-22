from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.fixture
def mgr():
    return create_mgr('a: f8; b: object; c: f8; d: object; e: f8;f: bool; g: i8; h: complex; i: datetime-1; j: datetime-2;k: M8[ns, US/Eastern]; l: M8[ns, CET];')