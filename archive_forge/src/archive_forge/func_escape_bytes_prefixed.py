import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_bytes_prefixed(value, mapping=None):
    return "_binary'%s'" % value.decode('ascii', 'surrogateescape').translate(_escape_table)