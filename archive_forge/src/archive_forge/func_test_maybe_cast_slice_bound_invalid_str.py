from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_maybe_cast_slice_bound_invalid_str(self, tdi):
    msg = 'cannot do slice indexing on TimedeltaIndex with these indexers \\[foo\\] of type str'
    with pytest.raises(TypeError, match=msg):
        tdi._maybe_cast_slice_bound('foo', side='left')
    with pytest.raises(TypeError, match=msg):
        tdi.get_slice_bound('foo', side='left')
    with pytest.raises(TypeError, match=msg):
        tdi.slice_locs('foo', None, None)