import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def parse_etag_response(value, strong=False):
    """
    Parse a response ETag.
    See:
        * http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.19
        * http://www.w3.org/Protocols/rfc2616/rfc2616-sec3.html#sec3.11
    """
    if not value:
        return None
    m = _rx_etag.match(value)
    if not m:
        return value
    elif strong and m.group(1):
        return None
    else:
        return m.group(2).replace('\\"', '"')