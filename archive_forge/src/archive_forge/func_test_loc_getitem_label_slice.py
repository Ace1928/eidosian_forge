from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('slc, typs, axes, fails', [[slice(1, 3), ['labels', 'mixed', 'empty', 'ts', 'floats'], None, TypeError], [slice('20130102', '20130104'), ['ts'], 1, TypeError], [slice(2, 8), ['mixed'], 0, TypeError], [slice(2, 8), ['mixed'], 1, KeyError], [slice(2, 4, 2), ['mixed'], 0, TypeError]])
@pytest.mark.parametrize('kind', ['series', 'frame'])
def test_loc_getitem_label_slice(self, slc, typs, axes, fails, kind, request):
    for typ in typs:
        obj = request.getfixturevalue(f'{kind}_{typ}')
        check_indexing_smoketest_or_raises(obj, 'loc', slc, axes=axes, fails=fails)