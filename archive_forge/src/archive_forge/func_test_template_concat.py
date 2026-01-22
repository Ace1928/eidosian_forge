import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_template_concat():
    u = Grid(0, 1)
    t1 = TuningParametersTemplate(dict(a=1, b=u, c=Grid(2, 3)))
    t2 = TuningParametersTemplate(dict(d=2, e=u, f=Grid(2, 3)))
    t = t1.concat(t2)
    assert dict(a=1, b=u, c=Grid(2, 3), d=2, e=u, f=Grid(2, 3)) == t
    assert dict(a=1, b=0, c=2) == t1.fill([0, 2])
    assert dict(d=2, e=1, f=3) == t2.fill([1, 3])
    assert dict(a=1, b=1, c=2, d=2, e=1, f=3) == t.fill([1, 2, 3])
    raises(ValueError, lambda: t.concat(t1))