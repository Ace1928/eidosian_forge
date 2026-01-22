from math import cos, pi, sin, tan
import numpy as np
import shapely
def interpret_origin(geom, origin, ndim):
    """Returns interpreted coordinate tuple for origin parameter.

    This is a helper function for other transform functions.

    The point of origin can be a keyword 'center' for the 2D bounding box
    center, 'centroid' for the geometry's 2D centroid, a Point object or a
    coordinate tuple (x0, y0, z0).
    """
    if origin == 'center':
        minx, miny, maxx, maxy = geom.bounds
        origin = ((maxx + minx) / 2.0, (maxy + miny) / 2.0)
    elif origin == 'centroid':
        origin = geom.centroid.coords[0]
    elif isinstance(origin, str):
        raise ValueError(f"'origin' keyword {origin!r} is not recognized")
    elif getattr(origin, 'geom_type', None) == 'Point':
        origin = origin.coords[0]
    if len(origin) not in (2, 3):
        raise ValueError("Expected number of items in 'origin' to be either 2 or 3")
    if ndim == 2:
        return origin[0:2]
    elif len(origin) == 2:
        return origin + (0.0,)
    else:
        return origin