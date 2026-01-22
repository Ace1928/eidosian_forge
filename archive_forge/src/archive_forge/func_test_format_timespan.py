import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
def test_format_timespan(self):
    """Test :func:`humanfriendly.format_timespan()`."""
    minute = 60
    hour = minute * 60
    day = hour * 24
    week = day * 7
    year = week * 52
    assert '1 nanosecond' == format_timespan(1e-09, detailed=True)
    assert '500 nanoseconds' == format_timespan(5e-07, detailed=True)
    assert '1 microsecond' == format_timespan(1e-06, detailed=True)
    assert '500 microseconds' == format_timespan(0.0005, detailed=True)
    assert '1 millisecond' == format_timespan(0.001, detailed=True)
    assert '500 milliseconds' == format_timespan(0.5, detailed=True)
    assert '0.5 seconds' == format_timespan(0.5, detailed=False)
    assert '0 seconds' == format_timespan(0)
    assert '0.54 seconds' == format_timespan(0.54321)
    assert '1 second' == format_timespan(1)
    assert '3.14 seconds' == format_timespan(math.pi)
    assert '1 minute' == format_timespan(minute)
    assert '1 minute and 20 seconds' == format_timespan(80)
    assert '2 minutes' == format_timespan(minute * 2)
    assert '1 hour' == format_timespan(hour)
    assert '2 hours' == format_timespan(hour * 2)
    assert '1 day' == format_timespan(day)
    assert '2 days' == format_timespan(day * 2)
    assert '1 week' == format_timespan(week)
    assert '2 weeks' == format_timespan(week * 2)
    assert '1 year' == format_timespan(year)
    assert '2 years' == format_timespan(year * 2)
    assert '6 years, 5 weeks, 4 days, 3 hours, 2 minutes and 500 milliseconds' == format_timespan(year * 6 + week * 5 + day * 4 + hour * 3 + minute * 2 + 0.5, detailed=True)
    assert '1 year, 2 weeks and 3 days' == format_timespan(year + week * 2 + day * 3 + hour * 12)
    assert '1 minute, 1 second and 100 milliseconds' == format_timespan(61.1, detailed=True)
    assert '1 minute and 1.1 seconds' == format_timespan(61.1, detailed=False)
    assert '1 minute and 0.3 seconds' == format_timespan(60.3)
    assert '5 minutes and 0.3 seconds' == format_timespan(300.3)
    assert '1 second and 15 milliseconds' == format_timespan(1.015, detailed=True)
    assert '10 seconds and 15 milliseconds' == format_timespan(10.015, detailed=True)
    assert '1 microsecond and 50 nanoseconds' == format_timespan(1.05e-06, detailed=True)
    now = datetime.datetime.now()
    then = now - datetime.timedelta(hours=23)
    assert '23 hours' == format_timespan(now - then)