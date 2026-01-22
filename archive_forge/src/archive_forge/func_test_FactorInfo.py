from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
def test_FactorInfo():
    fi1 = FactorInfo('asdf', 'numerical', {'a': 1}, num_columns=10)
    assert fi1.factor == 'asdf'
    assert fi1.state == {'a': 1}
    assert fi1.type == 'numerical'
    assert fi1.num_columns == 10
    assert fi1.categories is None
    repr(fi1)
    fi2 = FactorInfo('asdf', 'categorical', {'a': 2}, categories=['z', 'j'])
    assert fi2.factor == 'asdf'
    assert fi2.state == {'a': 2}
    assert fi2.type == 'categorical'
    assert fi2.num_columns is None
    assert fi2.categories == ('z', 'j')
    repr(fi2)
    import pytest
    pytest.raises(ValueError, FactorInfo, 'asdf', 'non-numerical', {})
    pytest.raises(ValueError, FactorInfo, 'asdf', 'numerical', {})
    pytest.raises(ValueError, FactorInfo, 'asdf', 'numerical', {}, num_columns='asdf')
    pytest.raises(ValueError, FactorInfo, 'asdf', 'numerical', {}, num_columns=1, categories=1)
    pytest.raises(TypeError, FactorInfo, 'asdf', 'categorical', {})
    pytest.raises(ValueError, FactorInfo, 'asdf', 'categorical', {}, num_columns=1)
    pytest.raises(TypeError, FactorInfo, 'asdf', 'categorical', {}, categories=1)
    if not six.PY3:
        fi_long = FactorInfo('asdf', 'numerical', {'a': 1}, num_columns=long(10))
        assert fi_long.num_columns == 10