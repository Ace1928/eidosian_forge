from __future__ import annotations
import re
import sys
from typing import Any, BinaryIO, List
from typing import Optional as OptionalType
from typing import TextIO, Tuple, Union
from pyparsing import CaselessKeyword as Keyword  # watch out :)
from pyparsing import (
import rdflib
from rdflib.compat import decodeUnicodeEscape
from . import operators as op
from .parserutils import Comp, CompValue, Param, ParamList
def expandUnicodeEscapes(q: str) -> str:
    """
    The syntax of the SPARQL Query Language is expressed over code points in Unicode [UNICODE]. The encoding is always UTF-8 [RFC3629].
    Unicode code points may also be expressed using an \\ uXXXX (U+0 to U+FFFF) or \\ UXXXXXXXX syntax (for U+10000 onwards) where X is a hexadecimal digit [0-9A-F]
    """

    def expand(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError) as e:
            raise ValueError('Invalid unicode code point: ' + m.group(1)) from e
    return expandUnicodeEscapes_re.sub(expand, q)