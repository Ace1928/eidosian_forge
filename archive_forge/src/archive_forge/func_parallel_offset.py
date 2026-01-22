import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry, JOIN_STYLE
from shapely.geometry.point import Point
def parallel_offset(self, distance, side='right', resolution=16, join_style=JOIN_STYLE.round, mitre_limit=5.0):
    """
        Alternative method to :meth:`offset_curve` method.

        Older alternative method to the :meth:`offset_curve` method, but uses
        ``resolution`` instead of ``quad_segs`` and a ``side`` keyword
        ('left' or 'right') instead of sign of the distance. This method is
        kept for backwards compatibility for now, but is is recommended to
        use :meth:`offset_curve` instead.
        """
    if side == 'right':
        distance *= -1
    return self.offset_curve(distance, quad_segs=resolution, join_style=join_style, mitre_limit=mitre_limit)