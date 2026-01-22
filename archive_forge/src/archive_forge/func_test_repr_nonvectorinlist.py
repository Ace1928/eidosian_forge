import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_repr_nonvectorinlist():
    vec = robjects.ListVector(OrderedDict((('a', 1), ('b', robjects.Formula('y ~ x')))))
    s = repr(vec).split(os.linesep)
    assert s[1].startswith("R classes: ('list',)")
    assert s[2].startswith('[IntSexpVector, LangSexpVector]')