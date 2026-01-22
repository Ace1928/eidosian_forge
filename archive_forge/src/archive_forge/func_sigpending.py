import _signal
from _signal import *
from enum import IntEnum as _IntEnum
@_wraps(_signal.sigpending)
def sigpending():
    return {_int_to_enum(x, Signals) for x in _signal.sigpending()}