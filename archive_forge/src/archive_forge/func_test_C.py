import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def test_C():
    c1 = C('asdf')
    assert isinstance(c1, _CategoricalBox)
    assert c1.data == 'asdf'
    assert c1.levels is None
    assert c1.contrast is None
    c2 = C('DATA', 'CONTRAST', 'LEVELS')
    assert c2.data == 'DATA'
    assert c2.contrast == 'CONTRAST'
    assert c2.levels == 'LEVELS'
    c3 = C(c2, levels='NEW LEVELS')
    assert c3.data == 'DATA'
    assert c3.contrast == 'CONTRAST'
    assert c3.levels == 'NEW LEVELS'
    c4 = C(c2, 'NEW CONTRAST')
    assert c4.data == 'DATA'
    assert c4.contrast == 'NEW CONTRAST'
    assert c4.levels == 'LEVELS'
    assert_no_pickling(c4)