from math import hypot, sqrt
from functools import wraps
from itertools import repeat
def igd(A, Z):
    """Inverse generational distance.
    """
    if not scipy_imported:
        raise ImportError('idg requires scipy module')
    distances = scipy.spatial.distance.cdist(A, Z)
    return numpy.average(numpy.min(distances, axis=0))