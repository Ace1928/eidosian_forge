import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
def eyring_equation(dH, dS, T, constants=None, units=None, backend=None):
    """
    Returns the rate coefficient according to the Eyring equation

    Parameters
    ----------
    dH: float with unit
        Enthalpy of activation.
    dS: float with unit
        Entropy of activation.
    T: float with unit
        temperature
    constants: object (optional, default: None)
        if None:
            T assumed to be in Kelvin, Ea in J/(K mol)
        else:
            attributes accessed: molar_gas_constant
            Tip: pass quantities.constants
    units: object (optional, default: None)
        attributes accessed: Joule, Kelvin and mol
    backend: module (optional)
        module with "exp", default: numpy, math

    """
    be = get_backend(backend)
    R = _get_R(constants, units)
    kB_over_h = _get_kB_over_h(constants, units)
    try:
        RT = (R * T).rescale(dH.dimensionality)
    except AttributeError:
        RT = R * T
    try:
        kB_over_h = kB_over_h.simplified
    except AttributeError:
        pass
    return kB_over_h * T * be.exp(dS / R) * be.exp(-dH / RT)