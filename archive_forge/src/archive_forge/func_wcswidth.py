from Markus Kuhn's C code, retrieved from:
from __future__ import division
import os
import sys
import warnings
from .table_wide import WIDE_EASTASIAN
from .table_zero import ZERO_WIDTH
from .unicode_versions import list_versions
def wcswidth(pwcs, n=None, unicode_version='auto'):
    """
    Given a unicode string, return its printable length on a terminal.

    :param str pwcs: Measure width of given unicode string.
    :param int n: When ``n`` is None (default), return the length of the
        entire string, otherwise width the first ``n`` characters specified.
    :param str unicode_version: An explicit definition of the unicode version
        level to use for determination, may be ``auto`` (default), which uses
        the Environment Variable, ``UNICODE_VERSION`` if defined, or the latest
        available unicode version, otherwise.
    :rtype: int
    :returns: The width, in cells, necessary to display the first ``n``
        characters of the unicode string ``pwcs``.  Returns ``-1`` if
        a non-printable character is encountered.
    """
    end = len(pwcs) if n is None else n
    idx = slice(0, end)
    width = 0
    for char in pwcs[idx]:
        wcw = wcwidth(char, unicode_version)
        if wcw < 0:
            return -1
        width += wcw
    return width