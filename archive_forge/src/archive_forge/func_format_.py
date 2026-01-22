import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.method
@specs.parameter('dt', yaqltypes.DateTime())
@specs.parameter('format__', yaqltypes.String())
def format_(dt, format__):
    """:yaql:format

    Returns a string representing datetime, controlled by a format string.

    :signature: dt.format(format)
    :receiverArg dt: input datetime object
    :argType dt: datetime object
    :arg format: format string
    :argType format: string
    :returnType: string

    .. code::

        yaql> now().format("%A, %d. %B %Y %I:%M%p")
        "Tuesday, 19. July 2016 08:49AM"
    """
    return dt.strftime(format__)