from webob.datetime_utils import (
from webob.descriptors import _rx_etag
from webob.util import header_docstring
def etag_property(key, default, rfc_section, strong=True):
    doc = header_docstring(key, rfc_section)
    doc += '  Converts it as a Etag.'

    def fget(req):
        value = req.environ.get(key)
        if not value:
            return default
        else:
            return ETagMatcher.parse(value, strong=strong)

    def fset(req, val):
        if val is None:
            req.environ[key] = None
        else:
            req.environ[key] = str(val)

    def fdel(req):
        del req.environ[key]
    return property(fget, fset, fdel, doc=doc)