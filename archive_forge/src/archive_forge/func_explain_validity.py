import shapely
def explain_validity(ob):
    """
    Explain the validity of the input geometry, if it is invalid.
    This will describe why the geometry is invalid, and might
    include a location if there is a self-intersection or a
    ring self-intersection.

    Parameters
    ----------
    ob: Geometry
        A shapely geometry object

    Returns
    -------
    str
        A string describing the reason the geometry is invalid.

    """
    return shapely.is_valid_reason(ob)