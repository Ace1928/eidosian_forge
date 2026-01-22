import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('timespan')
@specs.parameter('days', int)
@specs.parameter('hours', int)
@specs.parameter('minutes', int)
@specs.parameter('seconds', yaqltypes.Integer())
@specs.parameter('milliseconds', yaqltypes.Integer())
@specs.parameter('microseconds', yaqltypes.Integer())
def build_timespan(days=0, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0):
    """:yaql:timespan

    Returns timespan object with specified args.

    :signature: timespan(days => 0, hours => 0, minutes => 0, seconds => 0,
                         milliseconds => 0, microseconds => 0)
    :arg days: number of days in timespan, 0 by default
    :argType days: integer
    :arg hours: number of hours in timespan, 0 by default
    :argType hours: integer
    :arg minutes: number of minutes in timespan, 0 by default
    :argType minutes: integer
    :arg seconds: number of seconds in timespan, 0 by default
    :argType seconds: integer
    :arg milliseconds: number of microseconds in timespan, 0 by default
    :argType milliseconds: integer
    :arg microsecond: number of microseconds in timespan, 0 by default
    :argType microsecond: integer
    :returnType: timespan object

    .. code::

        yaql> timespan(days => 1, hours => 2, minutes => 3).hours
        26.05
    """
    return TIMESPAN_TYPE(days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds, microseconds=microseconds)