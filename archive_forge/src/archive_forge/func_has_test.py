from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
def has_test(combo):
    klass, dtype, method = combo
    cls_funcs = request.node.session.items
    return any((klass in x.name and dtype in x.name and (method in x.name) for x in cls_funcs))