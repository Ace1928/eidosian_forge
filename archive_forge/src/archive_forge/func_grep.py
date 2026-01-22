import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def grep(self, pattern, prune=False, field=None):
    """ Return all strings matching 'pattern' (a regex or callable)

        This is case-insensitive. If prune is true, return all items
        NOT matching the pattern.

        If field is specified, the match must occur in the specified
        whitespace-separated field.

        Examples::

            a.grep( lambda x: x.startswith('C') )
            a.grep('Cha.*log', prune=1)
            a.grep('chm', field=-1)
        """

    def match_target(s):
        if field is None:
            return s
        parts = s.split()
        try:
            tgt = parts[field]
            return tgt
        except IndexError:
            return ''
    if isinstance(pattern, str):
        pred = lambda x: re.search(pattern, x, re.IGNORECASE)
    else:
        pred = pattern
    if not prune:
        return SList([el for el in self if pred(match_target(el))])
    else:
        return SList([el for el in self if not pred(match_target(el))])