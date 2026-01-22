from __future__ import annotations
import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent
def strlines(s, c=64, short=False):
    """Return a cut-and-pastable string that, when printed, is
    equivalent to the input.  The lines will be surrounded by
    parentheses and no line will be longer than c (default 64)
    characters. If the line contains newlines characters, the
    `rawlines` result will be returned.  If ``short`` is True
    (default is False) then if there is one line it will be
    returned without bounding parentheses.

    Examples
    ========

    >>> from sympy.utilities.misc import strlines
    >>> q = 'this is a long string that should be broken into shorter lines'
    >>> print(strlines(q, 40))
    (
    'this is a long string that should be b'
    'roken into shorter lines'
    )
    >>> q == (
    ... 'this is a long string that should be b'
    ... 'roken into shorter lines'
    ... )
    True

    See Also
    ========
    filldedent, rawlines
    """
    if not isinstance(s, str):
        raise ValueError('expecting string input')
    if '\n' in s:
        return rawlines(s)
    q = '"' if repr(s).startswith('"') else "'"
    q = (q,) * 2
    if '\\' in s:
        m = '(\nr%s%%s%s\n)' % q
        j = '%s\nr%s' % q
        c -= 3
    else:
        m = '(\n%s%%s%s\n)' % q
        j = '%s\n%s' % q
        c -= 2
    out = []
    while s:
        out.append(s[:c])
        s = s[c:]
    if short and len(out) == 1:
        return (m % out[0]).splitlines()[1]
    return m % j.join(out)