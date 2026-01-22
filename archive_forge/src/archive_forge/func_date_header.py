import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def date_header(header, rfc_section):
    return converter_date(header_getter(header, rfc_section))