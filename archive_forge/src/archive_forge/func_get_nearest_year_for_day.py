import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def get_nearest_year_for_day(day):
    """
    Returns the nearest year to now inferred from a Julian date.

    >>> freezer = getfixture('freezer')
    >>> freezer.move_to('2019-05-20')
    >>> get_nearest_year_for_day(20)
    2019
    >>> get_nearest_year_for_day(340)
    2018
    >>> freezer.move_to('2019-12-15')
    >>> get_nearest_year_for_day(20)
    2020
    """
    now = time.gmtime()
    result = now.tm_year
    if day - now.tm_yday > 365 // 2:
        result -= 1
    if now.tm_yday - day > 365 // 2:
        result += 1
    return result