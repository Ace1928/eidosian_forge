from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_slice_subshape_negative_step():
    ds = 30 * Record([('a', 'int32')])
    assert ds.subshape[:-1] == 29 * Record([('a', 'int32')])