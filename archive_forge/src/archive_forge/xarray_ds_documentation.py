
    Assumes that the velocity components are given on a regular grid
    (fixed spacing in latitude and longitude).

    Parameters
    ----------
    u_var : str
        Name of the U-component (zonal) variable.
    v_var : str
        Name of the V-component (meridional) variable.
    lat_dim : str, optional
        Name of the latitude dimension/coordinate
        (default: 'latitude').
    lon_dim : str, optional
        Name of the longitude dimension/coordinate
        (default: 'longitude').
    units : str, optional
        Velocity units (default: try getting units from the
        'units' attributes of `u_var` and `v_var`).
    