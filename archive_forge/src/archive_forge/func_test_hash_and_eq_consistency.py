from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.parametrize('a,b', equiv_dshape_pairs)
def test_hash_and_eq_consistency(a, b):
    assert a == b
    assert hash(a) == hash(b)