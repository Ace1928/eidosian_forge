import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def build_meta_message(meta_type, data, delta=0):
    try:
        spec = _META_SPECS[meta_type]
    except KeyError:
        return UnknownMetaMessage(meta_type, data)
    else:
        msg = MetaMessage(spec.type, time=delta)
        spec.decode(msg, data)
        return msg