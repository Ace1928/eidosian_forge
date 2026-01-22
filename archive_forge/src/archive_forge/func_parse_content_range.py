import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def parse_content_range(value):
    if not value or not value.strip():
        return None
    return ContentRange.parse(value)