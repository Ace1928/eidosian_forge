import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
@pytest.mark.parametrize('bad_version', [(1, 2), (1,), [], '12', '123'])
def test_maybe_adjust_name_bad_version_raises(bad_version):
    msg = 'Version is incorrect, expected sequence of 3 integers'
    with pytest.raises(ValueError, match=msg):
        _maybe_adjust_name('values_block_0', version=bad_version)