from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def concat_codes_or_none(list_of_codes_or_none: list[cpy.CodeArray | None]) -> cpy.CodeArray | None:
    """Concatenate a list of codes arrays or None into a single code array or None.
    """
    list_of_codes = [codes for codes in list_of_codes_or_none if codes is not None]
    if list_of_codes:
        return concat_codes(list_of_codes)
    else:
        return None