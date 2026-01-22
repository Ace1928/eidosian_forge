from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_map_parse():
    result = dshape('var * {b: map[int32, {a: int64}]}')
    recmeasure = Map(dshape(int32), DataShape(Record([('a', int64)])))
    assert result == DataShape(var, Record([('b', recmeasure)]))