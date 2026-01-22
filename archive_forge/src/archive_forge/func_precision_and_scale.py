from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
def precision_and_scale(numeric_value):
    sign, digits, exp = numeric_value.as_tuple()
    number = 0
    for digit in digits:
        number = number * 10 + digit
    delta = 1
    number = 10 ** delta * number
    inumber = int(number)
    bits_req = inumber.bit_length() + 1
    bytes_req = (bits_req + 8) // 8
    if sign:
        inumber = -inumber
    prec = int(math.ceil(math.log10(abs(inumber))))
    scale = abs(exp)
    return (prec, scale, bytes_req, inumber)