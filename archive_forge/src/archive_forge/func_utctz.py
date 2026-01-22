import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
def utctz():
    """:yaql:utctz

    Returns UTC time zone in timespan object.

    :signature: utctz()
    :returnType: timespan object

    .. code::

        yaql> utctz().hours
        0.0
    """
    return ZERO_TIMESPAN