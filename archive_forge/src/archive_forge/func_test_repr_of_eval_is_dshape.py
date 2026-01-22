from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.parametrize('ds', ['int64', 'var * float64', '10 * var * int16', '{a: int32, b: ?string}', 'var * {a: int32, b: ?string}', '10 * {a: ?int32, b: var * {c: string[30]}}', '{"weird col": 3 * var * 2 * ?{a: int8, b: ?uint8}}', 'var * {"func-y": (A) -> var * {a: 10 * float64}}', 'decimal[18]', 'var * {amount: ?decimal[9,2]}'])
def test_repr_of_eval_is_dshape(ds):
    assert eval(repr(dshape(ds))) == dshape(ds)