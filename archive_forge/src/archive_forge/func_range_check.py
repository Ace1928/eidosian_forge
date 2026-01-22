import calendar
from collections import namedtuple
from aniso8601.exceptions import (
def range_check(valuestr, limit):
    if valuestr is None:
        return None
    if '.' in valuestr:
        castfunc = float
    else:
        castfunc = int
    value = cast(valuestr, castfunc, thrownmessage=limit.casterrorstring)
    if limit.min is not None and value < limit.min:
        raise limit.rangeexception(limit.rangeerrorstring)
    if limit.max is not None and value > limit.max:
        raise limit.rangeexception(limit.rangeerrorstring)
    return value