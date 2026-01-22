import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_warning_replaced_param(self):
    match = '.*`old[0,1]` is deprecated since version 0\\.10 and will be removed in 0\\.12.* see the documentation of .*_func_replace_params`.'
    with pytest.warns(FutureWarning, match=match):
        assert _func_replace_params(1, 2) == (1, DEPRECATED, DEPRECATED, None, 2, None)
    with pytest.warns(FutureWarning, match=match) as records:
        assert _func_replace_params(1, 2, 3) == (1, DEPRECATED, DEPRECATED, 3, 2, None)
    assert len(records) == 2
    assert '`old1` is deprecated' in records[0].message.args[0]
    assert '`old0` is deprecated' in records[1].message.args[0]
    with pytest.warns(FutureWarning, match=match):
        assert _func_replace_params(1, old0=2) == (1, DEPRECATED, DEPRECATED, None, 2, None)
    with pytest.warns(FutureWarning, match=match):
        assert _func_replace_params(1, old1=3) == (1, DEPRECATED, DEPRECATED, 3, None, None)
    with warnings.catch_warnings(record=True) as record:
        assert _func_replace_params(1, new0=2, new1=3) == (1, DEPRECATED, DEPRECATED, 2, 3, None)
    assert len(record) == 0