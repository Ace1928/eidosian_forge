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
def from_authority(cls, auth_name: str, code: Union[str, int]) -> 'CRS':
    """
        .. versionadded:: 2.2.0

        Make a CRS from an authority name and authority code

        Parameters
        ----------
        auth_name: str
            The name of the authority.
        code : int or str
            The code used by the authority.

        Returns
        -------
        CRS
        """
    return cls.from_user_input(_prepare_from_authority(auth_name, code))