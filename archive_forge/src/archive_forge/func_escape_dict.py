import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_dict(val, charset, mapping=None):
    n = {}
    for k, v in val.items():
        quoted = escape_item(v, charset, mapping)
        n[k] = quoted
    return n