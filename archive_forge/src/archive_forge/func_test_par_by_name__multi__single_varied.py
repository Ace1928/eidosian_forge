from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
import pytest
import numpy as np
from .. import ODESys, OdeSys, chained_parameter_variation  # OdeSys deprecated
from ..core import integrate_chained
from ..util import requires, pycvodes_klu
@requires('pygslodeiv2')
def test_par_by_name__multi__single_varied():
    ny = 3
    odesys1 = ODESys(*decay_factory(ny), param_names='a b c'.split(), par_by_name=True)
    params1 = {'a': 2, 'b': (3, 4, 5, 6, 7), 'c': 0}
    init1 = [42, 0, 0]
    results1 = odesys1.integrate(2.1, init1, params1, integrator='gsl')
    for idx1 in range(len(params1['b'])):
        ref_a1 = init1[0] * np.exp(-params1['a'] * results1[idx1].xout)
        ref_b1 = init1[0] * params1['a'] * (np.exp(-params1['a'] * results1[idx1].xout) - np.exp(-params1['b'][idx1] * results1[idx1].xout)) / (params1['b'][idx1] - params1['a'])
        ref_c1 = init1[0] - ref_a1 - ref_b1
        assert np.allclose(results1[idx1].yout[:, 0], ref_a1)
        assert np.allclose(results1[idx1].yout[:, 1], ref_b1)
        assert np.allclose(results1[idx1].yout[:, 2], ref_c1)
    odesys2 = ODESys(*decay_factory(ny), names='a b c'.split(), dep_by_name=True)
    init2 = {'a': (7, 13, 19, 23, 42, 101), 'b': 0, 'c': 0}
    params2 = [11.7, 12.3, 0]
    results2 = odesys2.integrate(3.4, init2, params2, integrator='gsl')
    for idx2 in range(len(init2['a'])):
        ref_a2 = init2['a'][idx2] * np.exp(-params2[0] * results2[idx2].xout)
        ref_b2 = init2['a'][idx2] * params2[0] * (np.exp(-params2[0] * results2[idx2].xout) - np.exp(-params2[1] * results2[idx2].xout)) / (params2[1] - params2[0])
        ref_c2 = init2['a'][idx2] - ref_a2 - ref_b2
        assert np.allclose(results2[idx2].yout[:, 0], ref_a2)
        assert np.allclose(results2[idx2].yout[:, 1], ref_b2)
        assert np.allclose(results2[idx2].yout[:, 2], ref_c2)