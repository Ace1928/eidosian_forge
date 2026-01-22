from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
def camel2title(strings: List[str], sep: str=' ', acronyms: Optional[List[str]]=None) -> List[str]:
    if isinstance(strings, str) or not hasattr(strings, '__iter__'):
        raise TypeError("camel2title() 'strings' argument must be iterable of strings")
    if len(strings) == 0:
        return strings
    if not isinstance(strings[0], str):
        raise TypeError("camel2title() 'strings' argument must be iterable of strings")
    if not isinstance(sep, str) or len(sep) != 1:
        raise ValueError(f"camel2title() 'sep' argument = '{sep}' must be single character")
    if _re.match('[a-zA-Z0-9]', sep):
        raise ValueError(f"camel2title() 'sep' argument = '{sep}' cannot be alpha-numeric")
    if _re.escape(sep) != sep and sep not in {' ', '-'}:
        raise ValueError(f"camel2title() 'sep' argument = '{sep}' cannot be special character")
    if acronyms is None:
        pat = '([a-z])([A-Z])'
        rep = f'\\g<1>{sep}\\g<2>'
        return [_re.sub(pat, rep, s).title() for s in strings]
    if isinstance(acronyms, str) or not hasattr(acronyms, '__iter__') or (not isinstance(acronyms[0], str)):
        raise TypeError("camel2title() 'acronyms' argument must be iterable of strings")
    for a in acronyms:
        if not _re.match('^[A-Z]+$', a):
            raise ValueError(f"camel2title() 'acronyms' argument must only contain upper-case, but '{a}' detected")
    pat = '([a-z])([A-Z])'
    rep = f'\\g<1>{sep}\\g<2>'
    strings = [_re.sub(pat, rep, s) for s in strings]
    for a in acronyms:
        pat = f'({a})([A-Z][a-z])'
        rep = f'\\g<1>{sep}\\g<2>'
        strings = [_re.sub(pat, rep, s) for s in strings]
    strings = [s.split(sep) for s in strings]
    strings = [[j.title() if j not in acronyms else j for j in s] for s in strings]
    strings = [sep.join(s) for s in strings]
    return strings