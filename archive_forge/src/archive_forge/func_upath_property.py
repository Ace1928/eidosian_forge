import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def upath_property(key):
    if PY2:

        def fget(req):
            encoding = req.url_encoding
            return req.environ.get(key, '').decode(encoding)

        def fset(req, val):
            encoding = req.url_encoding
            if isinstance(val, text_type):
                val = val.encode(encoding)
            req.environ[key] = val
    else:

        def fget(req):
            encoding = req.url_encoding
            return req.environ.get(key, '').encode('latin-1').decode(encoding)

        def fset(req, val):
            encoding = req.url_encoding
            req.environ[key] = val.encode(encoding).decode('latin-1')
    return property(fget, fset, doc='upath_property(%r)' % key)