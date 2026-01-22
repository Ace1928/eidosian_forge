from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
def postprocessor(data: str) -> str | float | bool | None:
    """
    Helper function to post process the results of the pattern matching functions in Cp2kOutput
    and turn them to Python types.

    Args:
        data (str): The data to be post processed.

    Raises:
        ValueError: If the data cannot be parsed.

    Returns:
        str | float | bool | None: The post processed data.
    """
    data = data.strip().replace(' ', '_')
    if data.lower() in {'false', 'no', 'f'}:
        return False
    if data.lower() == 'none':
        return None
    if data.lower() in {'true', 'yes', 't'}:
        return True
    if re.match('^-?\\d+$', data):
        try:
            return int(data)
        except ValueError as exc:
            raise ValueError(f'Error parsing {data!r} as int in CP2K file.') from exc
    if re.match('^[+\\-]?(?=.)(?:0|[1-9]\\d*)?(?:\\.\\d*)?(?:\\d[eE][+\\-]?\\d+)?$', data):
        try:
            return float(data)
        except ValueError as exc:
            raise ValueError(f'Error parsing {data!r} as float in CP2K file.') from exc
    if re.match('\\*+', data):
        return np.nan
    return data