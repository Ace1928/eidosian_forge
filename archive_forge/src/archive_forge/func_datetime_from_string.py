import datetime
import time as python_time
from yaql.language import specs
from yaql.language import yaqltypes
from dateutil import parser
from dateutil import tz
@specs.name('datetime')
@specs.parameter('string', yaqltypes.String())
@specs.parameter('format__', yaqltypes.String(True))
def datetime_from_string(string, format__=None):
    """:yaql:datetime

    Returns datetime object built by string parsed with format.

    :signature: datetime(string, format => null)
    :arg string: string representing datetime
    :argType string: string
    :arg format: format for parsing input string which should be supported
        with C99 standard of format codes. null by default, which means
        parsing with Python dateutil.parser usage
    :argType format: string
    :returnType: datetime object

    .. code::

        yaql> let(datetime("29.8?2015")) -> [$.year, $.month, $.day]
        [2015, 8, 29]
        yaql> let(datetime("29.8?2015", "%d.%m?%Y"))->[$.year, $.month, $.day]
        [2015, 8, 29]
    """
    if not format__:
        result = parser.parse(string)
    else:
        result = DATETIME_TYPE.strptime(string, format__)
    if not result.tzinfo:
        return result.replace(tzinfo=UTCTZ)
    return result