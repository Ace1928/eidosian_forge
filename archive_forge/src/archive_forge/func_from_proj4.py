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
def from_proj4(cls, in_proj_string: str) -> 'CRS':
    """
        .. versionadded:: 2.2.0

        Make a CRS from a PROJ string

        Parameters
        ----------
        in_proj_string : str
            A PROJ string.

        Returns
        -------
        CRS
        """
    if not is_proj(in_proj_string):
        raise CRSError(f'Invalid PROJ string: {in_proj_string}')
    return cls.from_user_input(_prepare_from_proj_string(in_proj_string))