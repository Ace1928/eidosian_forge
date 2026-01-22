import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
def localtz():
    """:yaql:localtz

    Returns local time zone in timespan object.

    :signature: localtz()
    :returnType: timespan object

    .. code::

        yaql> localtz().hours
        3.0
    """
    if python_time.daylight:
        return TIMESPAN_TYPE(seconds=-python_time.altzone)
    else:
        return TIMESPAN_TYPE(seconds=-python_time.timezone)