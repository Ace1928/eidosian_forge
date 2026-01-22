from __future__ import absolute_import
import re
from operator import itemgetter
import decimal
from .compat import binary_type, text_type, string_types, integer_types, PY3
from .decoder import PosInf
from .raw_json import RawJSON
def encode_basestring(s, _PY3=PY3, _q=u'"'):
    """Return a JSON representation of a Python string

    """
    if _PY3:
        if isinstance(s, bytes):
            s = str(s, 'utf-8')
        elif type(s) is not str:
            s = str.__str__(s)
    elif isinstance(s, str) and HAS_UTF8.search(s) is not None:
        s = unicode(s, 'utf-8')
    elif type(s) not in (str, unicode):
        if isinstance(s, str):
            s = str.__str__(s)
        else:
            s = unicode.__getnewargs__(s)[0]

    def replace(match):
        return ESCAPE_DCT[match.group(0)]
    return _q + ESCAPE.sub(replace, s) + _q