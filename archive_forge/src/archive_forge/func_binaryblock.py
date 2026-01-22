from __future__ import annotations
import numpy as np
from . import imageglobals as imageglobals
from .batteryrunners import BatteryRunner
from .volumeutils import Recoder, endian_codes, native_code, pretty_mapping, swapped_code
@property
def binaryblock(self):
    """binary block of data as string

        Returns
        -------
        binaryblock : string
            string giving binary data block

        Examples
        --------
        >>> # Make default empty structure
        >>> wstr = WrapStruct()
        >>> len(wstr.binaryblock)
        2
        """
    return self._structarr.tobytes()