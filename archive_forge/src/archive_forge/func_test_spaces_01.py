from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_spaces_01(self):
    space_dshape = "{'Unique Key': ?int64}"
    assert space_dshape == str(dshape(space_dshape))