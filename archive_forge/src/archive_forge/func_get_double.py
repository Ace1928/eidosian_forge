import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def get_double(value: Union[SupportsFloat, str], xsd_version: str='1.0') -> float:
    if isinstance(value, str):
        value = collapse_white_spaces(value)
        if value in NUMERIC_INF_OR_NAN or (xsd_version != '1.0' and value == '+INF'):
            if value == 'NaN':
                return math.nan
        elif value.lower() in INVALID_NUMERIC:
            raise ValueError(f'invalid value {value!r} for xs:double/xs:float')
    elif math.isnan(value):
        return math.nan
    return float(value)