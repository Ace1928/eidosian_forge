import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def safeStr(self, obj):
    try:
        return str(obj)
    except UnicodeEncodeError:
        try:
            return str(obj).encode(FALLBACK_ENCODING, 'replace')
        except UnicodeEncodeError:
            return repr(obj)