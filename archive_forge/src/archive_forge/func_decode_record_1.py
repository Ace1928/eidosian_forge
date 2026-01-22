import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
@classmethod
def decode_record_1(cls, s):
    r = Record()
    parts = s.split(',')[1:]
    assert len(parts) == len(cls.fields)
    for part, field in zip(parts, cls._fields):
        f, decode = field
        setattr(r, f, decode(part))
    return r