from __future__ import annotations
from typing import Callable
import pytest
from itertools import product
from numpy.testing import assert_allclose, suppress_warnings
from scipy import special
from scipy.special import cython_special
@pytest.mark.parametrize('param', PARAMS, ids=IDS)
def test_cython_api(param):
    pyfunc, cyfunc, specializations, knownfailure = param
    if knownfailure:
        pytest.xfail(reason=knownfailure)
    max_params = max((len(spec) for spec in specializations))
    values = [set() for _ in range(max_params)]
    for typecodes in specializations:
        for j, v in enumerate(typecodes):
            values[j].add(v)
    seen = set()
    is_fused_code = [False] * len(values)
    for j, v in enumerate(values):
        vv = tuple(sorted(v))
        if vv in seen:
            continue
        is_fused_code[j] = len(v) > 1
        seen.add(vv)
    for typecodes in specializations:
        signature = [CYTHON_SIGNATURE_MAP[code] for j, code in enumerate(typecodes) if is_fused_code[j]]
        if signature:
            cy_spec_func = cyfunc[tuple(signature)]
        else:
            signature = None
            cy_spec_func = cyfunc
        pts = _generate_test_points(typecodes)
        for pt in pts:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                pyval = pyfunc(*pt)
                cyval = cy_spec_func(*pt)
            assert_allclose(cyval, pyval, err_msg=f'{pt} {typecodes} {signature}')