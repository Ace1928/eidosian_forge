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
def test_encode_recursion_max(self):

    class O2:
        member = 0

    class O1:
        member = 0
    decoded_input = O1()
    decoded_input.member = O2()
    decoded_input.member.member = decoded_input
    with pytest.raises(OverflowError, match='Maximum recursion level reached'):
        ujson.ujson_dumps(decoded_input)