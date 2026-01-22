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
def is_exact_same(self, other: Any) -> bool:
    """
        Check if the CRS objects are the exact same.

        Parameters
        ----------
        other: Any
            Check if the other CRS is the exact same to this object.
            If the other object is not a CRS, it will try to create one.
            On Failure, it will return False.

        Returns
        -------
        bool
        """
    try:
        other = CRS.from_user_input(other)
    except CRSError:
        return False
    return self._crs.is_exact_same(other._crs)