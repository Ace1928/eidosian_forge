from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.pandas
@pytest.mark.skipif(sys.platform == 'win32' and (not util.windows_has_tzdata()), reason='Timezone database is not installed on Windows')
def test_assume_timezone():
    ts_type = pa.timestamp('ns')
    timestamps = pd.to_datetime(['1970-01-01T00:00:59.123456789', '2000-02-29T23:23:23.999999999', '2033-05-18T03:33:20.000000000', '2020-01-01T01:05:05.001', '2019-12-31T02:10:10.002', '2019-12-30T03:15:15.003', '2009-12-31T04:20:20.004132', '2010-01-01T05:25:25.005321', '2010-01-03T06:30:30.006163', '2010-01-04T07:35:35.0', '2006-01-01T08:40:40.0', '2005-12-31T09:45:45.0', '2008-12-28T00:00:00.0', '2008-12-29T00:00:00.0', '2012-01-01T01:02:03.0'])
    nonexistent = pd.to_datetime(['2015-03-29 02:30:00', '2015-03-29 03:30:00'])
    ambiguous = pd.to_datetime(['2018-10-28 01:20:00', '2018-10-28 02:36:00', '2018-10-28 03:46:00'])
    ambiguous_array = pa.array(ambiguous, type=ts_type)
    nonexistent_array = pa.array(nonexistent, type=ts_type)
    for timezone in ['UTC', 'US/Central', 'Asia/Kolkata']:
        options = pc.AssumeTimezoneOptions(timezone)
        ta = pa.array(timestamps, type=ts_type)
        expected = timestamps.tz_localize(timezone)
        result = pc.assume_timezone(ta, options=options)
        assert result.equals(pa.array(expected))
        result = pc.assume_timezone(ta, timezone)
        assert result.equals(pa.array(expected))
        ta_zoned = pa.array(timestamps, type=pa.timestamp('ns', timezone))
        with pytest.raises(pa.ArrowInvalid, match='already have a timezone:'):
            pc.assume_timezone(ta_zoned, options=options)
    invalid_options = pc.AssumeTimezoneOptions('Europe/Brusselsss')
    with pytest.raises(ValueError, match='not found in timezone database'):
        pc.assume_timezone(ta, options=invalid_options)
    timezone = 'Europe/Brussels'
    options_nonexistent_raise = pc.AssumeTimezoneOptions(timezone)
    options_nonexistent_earliest = pc.AssumeTimezoneOptions(timezone, ambiguous='raise', nonexistent='earliest')
    options_nonexistent_latest = pc.AssumeTimezoneOptions(timezone, ambiguous='raise', nonexistent='latest')
    with pytest.raises(ValueError, match=f"Timestamp doesn't exist in timezone '{timezone}'"):
        pc.assume_timezone(nonexistent_array, options=options_nonexistent_raise)
    expected = pa.array(nonexistent.tz_localize(timezone, nonexistent='shift_forward'))
    result = pc.assume_timezone(nonexistent_array, options=options_nonexistent_latest)
    expected.equals(result)
    expected = pa.array(nonexistent.tz_localize(timezone, nonexistent='shift_backward'))
    result = pc.assume_timezone(nonexistent_array, options=options_nonexistent_earliest)
    expected.equals(result)
    options_ambiguous_raise = pc.AssumeTimezoneOptions(timezone)
    options_ambiguous_latest = pc.AssumeTimezoneOptions(timezone, ambiguous='latest', nonexistent='raise')
    options_ambiguous_earliest = pc.AssumeTimezoneOptions(timezone, ambiguous='earliest', nonexistent='raise')
    with pytest.raises(ValueError, match=f"Timestamp is ambiguous in timezone '{timezone}'"):
        pc.assume_timezone(ambiguous_array, options=options_ambiguous_raise)
    expected = ambiguous.tz_localize(timezone, ambiguous=[True, True, True])
    result = pc.assume_timezone(ambiguous_array, options=options_ambiguous_earliest)
    result.equals(pa.array(expected))
    expected = ambiguous.tz_localize(timezone, ambiguous=[False, False, False])
    result = pc.assume_timezone(ambiguous_array, options=options_ambiguous_latest)
    result.equals(pa.array(expected))