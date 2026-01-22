import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def fractional_range_check(conversion, valuestr, limit):
    if valuestr is None:
        return None
    if '.' in valuestr:
        castfunc = partial(_cast_to_fractional_component, conversion)
    else:
        castfunc = int
    value = cast(valuestr, castfunc, thrownmessage=limit.casterrorstring)
    if type(value) is FractionalComponent:
        tocheck = float(valuestr)
    else:
        tocheck = int(valuestr)
    if limit.min is not None and tocheck < limit.min:
        raise limit.rangeexception(limit.rangeerrorstring)
    if limit.max is not None and tocheck > limit.max:
        raise limit.rangeexception(limit.rangeerrorstring)
    return value