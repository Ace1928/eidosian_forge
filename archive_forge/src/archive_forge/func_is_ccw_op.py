import numpy as np
import shapely
def is_ccw_op(ring):
    return signed_area(ring) >= 0.0