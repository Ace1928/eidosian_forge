import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('#operator_*')
@specs.parameter('n', yaqltypes.Number())
@specs.parameter('ts', TIMESPAN_TYPE)
def num_by_timespan(n, ts):
    """:yaql:operator *

    Returns timespan object built on number multiplied by timespan.

    :signature: left * right
    :arg left: number to multiply timespan
    :argType left: number
    :arg right: timespan object
    :argType right: timespan object
    :returnType: timespan

    .. code::

        yaql> let(2 * timespan(hours => 24)) -> $.hours
        48.0
    """
    return TIMESPAN_TYPE(microseconds=microseconds(ts) * n)