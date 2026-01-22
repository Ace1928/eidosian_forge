from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
@pytest.mark.parametrize('klass', _writers.values())
def test_subclass_attr(klass):
    attrs_base = {name for name in dir(ExcelWriter) if not name.startswith('_')}
    attrs_klass = {name for name in dir(klass) if not name.startswith('_')}
    assert not attrs_base.symmetric_difference(attrs_klass)