import datetime
import numbers
import sys
from base64 import b64decode as base64decode
from base64 import b64encode as base64encode
from functools import partial
from inspect import getmro
from itertools import takewhile
from kombu.utils.encoding import bytes_to_str, safe_repr, str_to_bytes
def find_pickleable_exception(exc, loads=pickle.loads, dumps=pickle.dumps):
    """Find first pickleable exception base class.

    With an exception instance, iterate over its super classes (by MRO)
    and find the first super exception that's pickleable.  It does
    not go below :exc:`Exception` (i.e., it skips :exc:`Exception`,
    :class:`BaseException` and :class:`object`).  If that happens
    you should use :exc:`UnpickleableException` instead.

    Arguments:
        exc (BaseException): An exception instance.
        loads: decoder to use.
        dumps: encoder to use

    Returns:
        Exception: Nearest pickleable parent exception class
            (except :exc:`Exception` and parents), or if the exception is
            pickleable it will return :const:`None`.
    """
    exc_args = getattr(exc, 'args', [])
    for supercls in itermro(exc.__class__, unwanted_base_classes):
        try:
            superexc = supercls(*exc_args)
            loads(dumps(superexc))
        except Exception:
            pass
        else:
            return superexc