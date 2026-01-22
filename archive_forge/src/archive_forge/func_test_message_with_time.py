import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize(['source', 'message', 'is_long'], [('ABC', string.ascii_lowercase, False), ('ABCDEF', string.ascii_lowercase, False), ('ABC', string.ascii_lowercase * 3, True), ('ABC' * 10, string.ascii_lowercase, True), ('ABC', string.ascii_lowercase + '၈', False)])
@pytest.mark.parametrize(['time', 'time_str'], [(0.2, '   0.2s'), (20, '  20.0s'), (2000, '33.3min'), (20000, '333.3min')])
def test_message_with_time(source, message, is_long, time, time_str):
    out = _message_with_time(source, message, time)
    if is_long:
        assert len(out) > 70
    else:
        assert len(out) == 70
    assert out.startswith('[' + source + '] ')
    out = out[len(source) + 3:]
    assert out.endswith(time_str)
    out = out[:-len(time_str)]
    assert out.endswith(', total=')
    out = out[:-len(', total=')]
    assert out.endswith(message)
    out = out[:-len(message)]
    assert out.endswith(' ')
    out = out[:-1]
    if is_long:
        assert not out
    else:
        assert list(set(out)) == ['.']