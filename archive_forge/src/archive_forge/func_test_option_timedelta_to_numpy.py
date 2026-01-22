from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.parametrize('unit', ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'])
def test_option_timedelta_to_numpy(unit):
    assert Option(TimeDelta(unit=unit)).to_numpy_dtype() == np.dtype('timedelta64[%s]' % unit)