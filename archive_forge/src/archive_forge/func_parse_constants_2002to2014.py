from __future__ import annotations
import warnings
from typing import Any
def parse_constants_2002to2014(d: str) -> dict[str, tuple[float, str, float]]:
    constants = {}
    for line in d.split('\n'):
        name = line[:55].rstrip()
        val = float(line[55:77].replace(' ', '').replace('...', ''))
        uncert = float(line[77:99].replace(' ', '').replace('(exact)', '0'))
        units = line[99:].rstrip()
        constants[name] = (val, units, uncert)
    return constants