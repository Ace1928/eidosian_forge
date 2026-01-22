from functools import partial
import json
import math
import warnings
from fiona.model import Geometry, to_dict
from fiona._vendor.munch import munchify
def recursive_round(obj, precision):
    """Recursively round coordinates."""
    if precision < 0:
        return obj
    if getattr(obj, 'geometries', None):
        return Geometry(geometries=[recursive_round(part, precision) for part in obj.geometries])
    elif getattr(obj, 'coordinates', None):
        return Geometry(coordinates=[recursive_round(part, precision) for part in obj.coordinates])
    if isinstance(obj, (int, float)):
        return round(obj, precision)
    else:
        return [recursive_round(part, precision) for part in obj]