import _signal
from _signal import *
from enum import IntEnum as _IntEnum
@_wraps(_signal.valid_signals)
def valid_signals():
    return {_int_to_enum(x, Signals) for x in _signal.valid_signals()}