import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
@pytest.mark.skipif(not IS64, reason='not compliant on 32-bit, xref #15865')
def test_encode_decimal(self):
    sut = decimal.Decimal('1337.1337')
    encoded = ujson.ujson_dumps(sut, double_precision=15)
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 1337.1337
    sut = decimal.Decimal('0.95')
    encoded = ujson.ujson_dumps(sut, double_precision=1)
    assert encoded == '1.0'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 1.0
    sut = decimal.Decimal('0.94')
    encoded = ujson.ujson_dumps(sut, double_precision=1)
    assert encoded == '0.9'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 0.9
    sut = decimal.Decimal('1.95')
    encoded = ujson.ujson_dumps(sut, double_precision=1)
    assert encoded == '2.0'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 2.0
    sut = decimal.Decimal('-1.95')
    encoded = ujson.ujson_dumps(sut, double_precision=1)
    assert encoded == '-2.0'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == -2.0
    sut = decimal.Decimal('0.995')
    encoded = ujson.ujson_dumps(sut, double_precision=2)
    assert encoded == '1.0'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 1.0
    sut = decimal.Decimal('0.9995')
    encoded = ujson.ujson_dumps(sut, double_precision=3)
    assert encoded == '1.0'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 1.0
    sut = decimal.Decimal('0.99999999999999944')
    encoded = ujson.ujson_dumps(sut, double_precision=15)
    assert encoded == '1.0'
    decoded = ujson.ujson_loads(encoded)
    assert decoded == 1.0