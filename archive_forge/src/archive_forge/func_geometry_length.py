import math
import warnings
from typing import Any, Optional, Union
from pyproj._geod import Geod as _Geod
from pyproj._geod import GeodIntermediateReturn, geodesic_version_str
from pyproj._geod import reverse_azimuth as _reverse_azimuth
from pyproj.enums import GeodIntermediateFlag
from pyproj.exceptions import GeodError
from pyproj.list import get_ellps_map
from pyproj.utils import DataType, _convertback, _copytobuffer
def geometry_length(self, geometry, radians: bool=False) -> float:
    """
        .. versionadded:: 2.3.0

        Returns the geodesic length (meters) of the shapely geometry.

        If it is a Polygon, it will return the sum of the
        lengths along the perimeter.
        If it is a MultiPolygon or MultiLine, it will return
        the sum of the lengths.

        Example usage:

        >>> from pyproj import Geod
        >>> from shapely.geometry import Point, LineString
        >>> line_string = LineString([Point(1, 2), Point(3, 4)])
        >>> geod = Geod(ellps="WGS84")
        >>> f"{geod.geometry_length(line_string):.3f}"
        '313588.397'

        Parameters
        ----------
        geometry: :class:`shapely.geometry.BaseGeometry`
            The geometry to calculate the length from.
        radians: bool, default=False
            If True, the input data is assumed to be in radians.
            Otherwise, the data is assumed to be in degrees.

        Returns
        -------
        float:
            The total geodesic length of the geometry (meters).
        """
    try:
        return self.line_length(*geometry.xy, radians=radians)
    except (AttributeError, NotImplementedError):
        pass
    if hasattr(geometry, 'exterior'):
        return self.geometry_length(geometry.exterior, radians=radians)
    if hasattr(geometry, 'geoms'):
        total_length = 0.0
        for geom in geometry.geoms:
            total_length += self.geometry_length(geom, radians=radians)
        return total_length
    raise GeodError('Invalid geometry provided.')