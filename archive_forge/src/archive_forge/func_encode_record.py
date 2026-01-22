import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
@classmethod
def encode_record(cls, value):
    values = [cls.this_version]
    values.extend(['%s' % getattr(value, x) for x in cls.fields])
    return ','.join(values)