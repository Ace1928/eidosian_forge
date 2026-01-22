from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_string_to_tzinfo():
    string = ['UTC', 'Europe/Paris', '+03:00', '+01:30', '-02:00']
    try:
        import pytz
        expected = [pytz.utc, pytz.timezone('Europe/Paris'), pytz.FixedOffset(180), pytz.FixedOffset(90), pytz.FixedOffset(-120)]
        result = [pa.lib.string_to_tzinfo(i) for i in string]
        assert result == expected
    except ImportError:
        try:
            import zoneinfo
            expected = [zoneinfo.ZoneInfo(key='UTC'), zoneinfo.ZoneInfo(key='Europe/Paris'), datetime.timezone(datetime.timedelta(hours=3)), datetime.timezone(datetime.timedelta(hours=1, minutes=30)), datetime.timezone(-datetime.timedelta(hours=2))]
            result = [pa.lib.string_to_tzinfo(i) for i in string]
            assert result == expected
        except ImportError:
            pytest.skip('requires pytz or zoneinfo to be installed')