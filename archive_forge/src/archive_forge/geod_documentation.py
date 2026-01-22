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

        equality operator == for Geod objects

        Example usage:

        >>> from pyproj import Geod
        >>> # Use Clarke 1866 ellipsoid.
        >>> gclrk1 = Geod(ellps='clrk66')
        >>> # Define Clarke 1866 using parameters
        >>> gclrk2 = Geod(a=6378206.4, b=6356583.8)
        >>> gclrk1 == gclrk2
        True
        >>> # WGS 66 ellipsoid, PROJ style
        >>> gwgs66 = Geod('+ellps=WGS66')
        >>> # Naval Weapons Lab., 1965 ellipsoid
        >>> gnwl9d = Geod('+ellps=NWL9D')
        >>> # these ellipsoids are the same
        >>> gnwl9d == gwgs66
        True
        >>> gclrk1 != gnwl9d  # Clarke 1866 is unlike NWL9D
        True
        