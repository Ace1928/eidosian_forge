import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def header_getter(header, rfc_section):
    doc = header_docstring(header, rfc_section)
    key = header.lower()

    def fget(r):
        for k, v in r._headerlist:
            if k.lower() == key:
                return v

    def fset(r, value):
        fdel(r)
        if value is not None:
            if '\n' in value or '\r' in value:
                raise ValueError('Header value may not contain control characters')
            if isinstance(value, text_type) and PY2:
                value = value.encode('latin-1')
            r._headerlist.append((header, value))

    def fdel(r):
        r._headerlist[:] = [(k, v) for k, v in r._headerlist if k.lower() != key]
    return property(fget, fset, fdel, doc)