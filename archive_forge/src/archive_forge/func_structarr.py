from __future__ import annotations
import numpy as np
from . import imageglobals as imageglobals
from .batteryrunners import BatteryRunner
from .volumeutils import Recoder, endian_codes, native_code, pretty_mapping, swapped_code
@property
def structarr(self):
    """Structured data, with data fields

        Examples
        --------
        >>> wstr1 = WrapStruct() # with default data
        >>> an_int = wstr1.structarr['integer']
        >>> wstr1.structarr = None
        Traceback (most recent call last):
           ...
        AttributeError: ...
        """
    return self._structarr