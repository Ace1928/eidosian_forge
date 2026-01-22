import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def raise_with_context(exc):
    exc_info = sys.exc_info()
    if not exc_info:
        raise exc
    elif exc_info[1] is exc:
        raise
    raise exc from exc_info[1]