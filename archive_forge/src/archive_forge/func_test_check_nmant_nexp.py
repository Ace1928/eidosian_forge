import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_check_nmant_nexp():
    for t in IEEE_floats:
        nmant = np.finfo(t).nmant
        maxexp = np.finfo(t).maxexp
        assert _check_nmant(t, nmant)
        assert not _check_nmant(t, nmant - 1)
        assert not _check_nmant(t, nmant + 1)
        with suppress_warnings():
            assert _check_maxexp(t, maxexp)
        assert not _check_maxexp(t, maxexp - 1)
        with suppress_warnings():
            assert not _check_maxexp(t, maxexp + 1)
    for t in ok_floats():
        ti = type_info(t)
        if ti['nmant'] not in (105, 106):
            assert _check_nmant(t, ti['nmant'])
        if t != np.longdouble or sys.platform != 'darwin':
            assert _check_maxexp(t, ti['maxexp'])