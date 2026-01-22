import numpy as np
import shapely
def is_ccw_impl(name=None):
    """Predicate implementation"""

    def is_ccw_op(ring):
        return signed_area(ring) >= 0.0
    if shapely.geos_version >= (3, 7, 0):
        return shapely.is_ccw
    else:
        return is_ccw_op