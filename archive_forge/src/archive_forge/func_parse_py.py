from __future__ import annotations
import re
import warnings
from traitlets.log import get_logger
from nbformat import v3 as _v_latest
from nbformat.v3 import (
from . import versions
from .converter import convert
from .reader import reads as reader_reads
from .validator import ValidationError, validate
def parse_py(s, **kwargs):
    """Parse a string into a (nbformat, string) tuple."""
    nbf = current_nbformat
    nbm = current_nbformat_minor
    pattern = '# <nbformat>(?P<nbformat>\\d+[\\.\\d+]*)</nbformat>'
    m = re.search(pattern, s)
    if m is not None:
        digits = m.group('nbformat').split('.')
        nbf = int(digits[0])
        if len(digits) > 1:
            nbm = int(digits[1])
    return (nbf, nbm, s)