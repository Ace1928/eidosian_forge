from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.xfail(raises=TypeError, reason='Not yet implemented')
def test_datetime_with_tz_not_a_string():
    assert DateTime(tz=datetime.tzinfo())