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
@classmethod
def from_wkt(cls, in_wkt_string: str) -> 'CRS':
    """
        .. versionadded:: 2.2.0

        Make a CRS from a WKT string

        Parameters
        ----------
        in_wkt_string : str
            A WKT string.

        Returns
        -------
        CRS
        """
    if not is_wkt(in_wkt_string):
        raise CRSError(f'Invalid WKT string: {in_wkt_string}')
    return cls.from_user_input(_prepare_from_string(in_wkt_string))