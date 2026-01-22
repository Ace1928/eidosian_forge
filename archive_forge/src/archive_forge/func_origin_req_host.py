import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
@property
def origin_req_host(self):
    return self.get_origin_req_host()