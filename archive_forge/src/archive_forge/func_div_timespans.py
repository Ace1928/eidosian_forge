import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_/')
@specs.parameter('ts1', TIMESPAN_TYPE)
@specs.parameter('ts2', TIMESPAN_TYPE)
def div_timespans(ts1, ts2):
    """:yaql:operator /

    Returns result of division of timespan microseconds by another timespan
    microseconds.

    :signature: left / right
    :arg left: left timespan object
    :argType left: timespan object
    :arg right: right timespan object
    :argType right: timespan object
    :returnType: float

    .. code::

        yaql> timespan(hours => 24) / timespan(hours => 12)
        2.0
    """
    return (0.0 + microseconds(ts1)) / microseconds(ts2)