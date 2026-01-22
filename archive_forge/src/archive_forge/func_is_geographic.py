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
@property
def is_geographic(self) -> bool:
    """
        This checks if the CRS is geographic.
        It will check if it has a geographic CRS
        in the sub CRS if it is a compound CRS and will check if
        the source CRS is geographic if it is a bound CRS.

        Returns
        -------
        bool:
            True if the CRS is in geographic (lon/lat) coordinates.
        """
    return self._crs.is_geographic