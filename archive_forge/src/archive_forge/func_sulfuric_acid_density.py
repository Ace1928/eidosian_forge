import numpy as np
import warnings
from ..util import NoConvergence
def sulfuric_acid_density(w, T=None, T0=None, units=None, warn=True):
    """
    Density of sulfuric acid (kg/m³) as function of temperature (K)
    and mass fraction acid (w).

    Parameters
    ----------
    w: float
        Acid mass fraction (0.1 <= w <= 0.9)
    T: float
        Temperature (in Kelvin) (273 <= T <= 323) (default: 298.15)
    T0: float
        Value of T for 0 degree Celsius (default: 273.15)
    units: object (optional)
        object with attributes: kelvin, meter, kilogram
    warn: bool (default: True)
        Emit UserWarning when outside T or w range.

    Returns
    -------
    Density of sulfuric acid (float of kg/m³ if T is float and units is None)

    Examples
    --------
    >>> print('%d' % sulfuric_acid_density(.5, 293))
    1396

    References
    ----------
    Cathrine E. L. Myhre , Claus J. Nielsen ,* and Ole W. Saastad
        "Density and Surface Tension of Aqueous H2SO4 at Low Temperature"
        J. Chem. Eng. Data, 1998, 43 (4), pp 617–622
        http://pubs.acs.org/doi/abs/10.1021/je980013g
        DOI: 10.1021/je980013g
    """
    if units is None:
        K = 1
        m = 1
        kg = 1
    else:
        K = units.Kelvin
        m = units.meter
        kg = units.kilogram
    if T is None:
        T = 298.15 * K
    m3 = m ** 3
    if T0 is None:
        T0 = 273.15 * K
    t = T - T0
    if warn:
        if np.any(t < 0 * K) or np.any(t > 50 * K):
            warnings.warn('Temperature is outside range (0-50 degC)')
        if np.any(w < 0.1) or np.any(w > 0.9):
            warnings.warn('Mass fraction is outside range (0.1-0.9)')
    t_arr = np.array([float(t / K) ** j for j in range(5)]).reshape((1, 5))
    w_arr = np.array([w ** i for i in range(11)]).reshape((11, 1))
    return np.sum(t_arr * w_arr * _data) * kg / m3