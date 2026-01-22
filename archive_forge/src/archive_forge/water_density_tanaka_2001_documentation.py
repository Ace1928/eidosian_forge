import warnings

    Density of water (kg/m3) as function of temperature (K)
    according to VSMOW model between 0 and 40 degree Celsius.
    Fitted using Thiesen's equation.

    Parameters
    ----------
    T : float
        Temperature (in Kelvin) (default: 298.15).
    T0 : float
        Value of T for 0 degree Celsius (default: 273.15).
    units : object (optional)
        Object with attributes: Kelvin, meter, kilogram.
    a : array_like (optional)
        5 parameters to the equation.
    just_return_a : bool (optional, default: False)
        Do not compute rho, just return the parameters ``a``.
    warn : bool (default: True)
        Emit UserWarning when outside temperature range.

    Returns
    -------
    Density of water (float of kg/m3 if T is float and units is None)

    Examples
    --------
    >>> print('%.2f' % water_density(277.13))
    999.97

    References
    ----------
    TANAKA M., GIRARD G., DAVIS R., PEUTO A. and BIGNELL N.,
        "Recommended table for the density of water between 0 °C and 40 °C
        based on recent experimental reports",
        Metrologia, 2001, 38, 301-309.
        http://iopscience.iop.org/article/10.1088/0026-1394/38/4/3
        doi:10.1088/0026-1394/38/4/3
    