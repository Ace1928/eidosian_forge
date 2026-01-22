import json
import re
import threading
import warnings
from typing import Any, Callable, Optional, Union
from pyproj._crs import (
from pyproj.crs._cf1x8 import (
from pyproj.crs.coordinate_operation import ToWGS84Transformation
from pyproj.crs.coordinate_system import Cartesian2DCS, Ellipsoidal2DCS, VerticalCS
from pyproj.enums import ProjVersion, WktVersion
from pyproj.exceptions import CRSError
from pyproj.geod import Geod
def get_geod(self) -> Optional[Geod]:
    """
        Returns
        -------
        pyproj.geod.Geod:
            Geod object based on the ellipsoid.
        """
    if self.ellipsoid is None:
        return None
    return Geod(a=self.ellipsoid.semi_major_metre, rf=self.ellipsoid.inverse_flattening, b=self.ellipsoid.semi_minor_metre)