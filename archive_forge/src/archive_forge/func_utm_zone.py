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
def utm_zone(self) -> Optional[str]:
    """
        .. versionadded:: 2.6.0

        Finds the UTM zone in a Projected CRS, Bound CRS, or Compound CRS

        Returns
        -------
        Optional[str]:
            The UTM zone number and letter if applicable.
        """
    if self.is_bound and self.source_crs:
        return self.source_crs.utm_zone
    if self.sub_crs_list:
        for sub_crs in self.sub_crs_list:
            if sub_crs.utm_zone:
                return sub_crs.utm_zone
    elif self.coordinate_operation and 'UTM ZONE' in self.coordinate_operation.name.upper():
        return self.coordinate_operation.name.upper().split('UTM ZONE ')[-1]
    return None