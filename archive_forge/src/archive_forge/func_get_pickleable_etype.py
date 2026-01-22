import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def get_pickleable_etype(cls, loads=pickle.loads, dumps=pickle.dumps):
    """Get pickleable exception type."""
    try:
        loads(dumps(cls))
    except Exception:
        return Exception
    else:
        return cls