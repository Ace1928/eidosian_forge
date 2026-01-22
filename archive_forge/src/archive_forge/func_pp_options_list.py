from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def pp_options_list(keys: Iterable[str], width: int=80, _print: bool=False):
    """Builds a concise listing of available options, grouped by prefix"""
    from itertools import groupby
    from textwrap import wrap

    def pp(name: str, ks: Iterable[str]) -> list[str]:
        pfx = '- ' + name + '.[' if name else ''
        ls = wrap(', '.join(ks), width, initial_indent=pfx, subsequent_indent='  ', break_long_words=False)
        if ls and ls[-1] and name:
            ls[-1] = ls[-1] + ']'
        return ls
    ls: list[str] = []
    singles = [x for x in sorted(keys) if x.find('.') < 0]
    if singles:
        ls += pp('', singles)
    keys = [x for x in keys if x.find('.') >= 0]
    for k, g in groupby(sorted(keys), lambda x: x[:x.rfind('.')]):
        ks = [x[len(k) + 1:] for x in list(g)]
        ls += pp(k, ks)
    s = '\n'.join(ls)
    if _print:
        print(s)
    else:
        return s