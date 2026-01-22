import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def clear_time_override():
    """Remove the overridden time.

    See :py:class:`oslo_utils.fixture.TimeFixture`.

    """
    utcnow.override_time = None