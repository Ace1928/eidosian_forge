from collections import defaultdict
from functools import reduce
from operator import mul
import pytest
from chempy import ReactionSystem
from chempy.units import (
from chempy.util.testing import requires
from ..integrated import binary_rev
from ..ode import get_odesys
from .._native import get_native
@requires('pycvodes', 'pyodesys')
@pytest.mark.parametrize('scaling_density', [(1, False), (763, False), (1, True)])
def test_get_native__Radiolytic__named_parameter__units(scaling_density):
    scaling, density = scaling_density
    rsys = ReactionSystem.from_string("\n    -> H; Radiolytic(2*per100eV)\n    H + H -> H2; 'k2'\n    ", checks=('substance_keys', 'duplicate', 'duplicate_names'))
    gval = 2 * u.per100eV
    from pyodesys.symbolic import ScaledSys
    kwargs = {} if scaling == 1 else dict(SymbolicSys=ScaledSys, dep_scaling=scaling)
    dens = {'density': 998 * u.g / u.dm3}
    odesys, extra = get_odesys(rsys, include_params=False, substitutions=dens if density else {}, unit_registry=SI_base_registry, **kwargs)
    c0 = {'H': 4.2e-05 * u.molar, 'H2': 17000.0 * u.nanomolar}
    native = get_native(rsys, odesys, 'cvode')
    tend = 7 * 60 * u.minute
    params = {'doserate': 314 * u.Gy / u.hour, 'k2': 53 / u.molar / u.minute}
    if not density:
        params.update(dens)
    result = native.integrate(tend, c0, params, atol=1e-15, rtol=1e-15, integrator='cvode', nsteps=8000)
    assert result.info['success']

    def analytic_H(t, p, k, H0):
        x0 = np.sqrt(2) * np.sqrt(p)
        x1 = x0
        x2 = np.sqrt(k)
        x3 = t * x1 * x2
        x4 = H0 * x2
        x5 = np.sqrt(x0 + 2 * x4)
        x6 = np.sqrt(-1 / (2 * H0 * x2 - x0))
        x7 = x5 * x6 * np.exp(x3)
        x8 = np.exp(-x3) / (x5 * x6)
        return x1 * (x7 - x8) / (2 * x2 * (x7 + x8))
    t_ul = to_unitless(result.xout, u.s)
    p_ul = to_unitless(params['doserate'] * dens['density'] * gval, u.micromolar / u.s)
    ref_H_uM = analytic_H(t_ul, p_ul, to_unitless(params['k2'], 1 / u.micromolar / u.s), to_unitless(c0['H'], u.micromolar))
    ref_H2_uM = to_unitless(c0['H2'], u.micromolar) + to_unitless(c0['H'], u.micromolar) / 2 + t_ul * p_ul / 2 - ref_H_uM / 2
    assert np.allclose(to_unitless(result.named_dep('H'), u.micromolar), ref_H_uM)
    assert np.allclose(to_unitless(result.named_dep('H2'), u.micromolar), ref_H2_uM)