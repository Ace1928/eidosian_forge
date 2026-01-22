import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def days_from_common_era(year: int) -> int:
    """
    Returns the number of days from 0001-01-01 to the provided year. For a
    common era year the days are counted until the last day of December, for a
    BCE year the days are counted down from the end to the 1st of January.
    """
    if year > 0:
        return year * 365 + year // 4 - year // 100 + year // 400
    elif year >= -1:
        return year * 366
    else:
        year = -year - 1
        return -(366 + year * 365 + year // 4 - year // 100 + year // 400)