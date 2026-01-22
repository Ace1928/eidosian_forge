from collections import namedtuple
import math
import warnings
@classmethod
def from_gdal(cls, c: float, a: float, b: float, f: float, d: float, e: float):
    """Use same coefficient order as GDAL's GetGeoTransform().

        :param c, a, b, f, d, e: 6 floats ordered by GDAL.
        :rtype: Affine
        """
    return cls.__new__(cls, a, b, c, d, e, f)