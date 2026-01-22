import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def serialize_content_range(value):
    if isinstance(value, (tuple, list)):
        if len(value) not in (2, 3):
            raise ValueError('When setting content_range to a list/tuple, it must be length 2 or 3 (not %r)' % value)
        if len(value) == 2:
            begin, end = value
            length = None
        else:
            begin, end, length = value
        value = ContentRange(begin, end, length)
    value = str(value).strip()
    if not value:
        return None
    return value