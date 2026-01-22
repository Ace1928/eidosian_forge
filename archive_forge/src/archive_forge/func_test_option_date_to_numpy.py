from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_option_date_to_numpy():
    assert Option(Date()).to_numpy_dtype() == np.dtype('datetime64[D]')