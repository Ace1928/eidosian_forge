from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_format_bug(self):
    now = datetime.now()
    msg = 'Index\\.format is deprecated'
    if not str(now).endswith('000'):
        index = Index([now])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = index.format()
        expected = [str(index[0])]
        assert formatted == expected
    with tm.assert_produces_warning(FutureWarning, match=msg):
        Index([]).format()