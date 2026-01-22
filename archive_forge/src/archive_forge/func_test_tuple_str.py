from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_tuple_str():
    assert str(Tuple([Option(int32), float64])) == '(?int32, float64)'