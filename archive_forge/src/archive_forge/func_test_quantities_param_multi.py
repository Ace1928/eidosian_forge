from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('quantities')
def test_quantities_param_multi():
    import quantities as pq

    def pp(x, y, p):
        return (x, y, np.array([[item.rescale(1 / pq.s).magnitude for item in p[:, idx]] for idx in range(1)]).T)
    odesys = ODESys(sine, sine_jac, param_names=['k'], par_by_name=True, pre_processors=[pp])
    A = 2.0
    kvals = (7452.0, 13853.0, 22123.0)
    results = odesys.integrate(np.linspace(0, 1), [[0, A * kval / 3600] for kval in kvals], {'k': [val / pq.hour for val in kvals]})
    assert len(results) == 3
    assert all([r.info['success'] for r in results])
    for res, kval in zip(results, kvals):
        assert res.xout.size > 7
        ref = [A * np.sin(1 / 3600 * kval * (res.xout - res.xout[0])), A * np.cos(1 / 3600 * kval * (res.xout - res.xout[0])) * kval / 3600]
        assert np.allclose(res.yout[:, 0], ref[0], atol=1e-05, rtol=1e-05)
        assert np.allclose(res.yout[:, 1], ref[1], atol=1e-05, rtol=1e-05)