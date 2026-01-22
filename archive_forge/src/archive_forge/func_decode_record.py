import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
@classmethod
def decode_record(cls, s):
    try:
        s = s.decode('utf8')
    except UnicodeError:
        raise StandardError("don't know how to handle db value %s" % repr(s))
    parts = s.split(',')
    version = parts[0]
    if len(parts) == 3:
        dispatch = cls.decode_record_0
    elif version == '1':
        dispatch = cls.decode_record_1
    else:
        raise StandardError("don't know how to handle db value %s" % repr(s))
    return dispatch(s)