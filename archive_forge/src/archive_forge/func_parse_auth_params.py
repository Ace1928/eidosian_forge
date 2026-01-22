import re
from datetime import (
from collections import namedtuple
from webob.byterange import (
from webob.compat import (
from webob.datetime_utils import (
from webob.util import (
def parse_auth_params(params):
    r = {}
    for k, v in _rx_auth_param.findall(params):
        r[k] = v.strip('"')
    return r