import datetime
from collections import namedtuple
from functools import partial
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
@classmethod
def range_check_interval(cls, start=None, end=None, duration=None):
    if start is not None and end is not None:
        if cls._is_interval_end_concise(end) is True:
            end = cls._combine_concise_interval_tuples(start, end)
        return (start, end, duration)
    durationobject = cls._build_object(duration)
    if end is not None:
        endobject = cls._build_object(end)
        if type(end) is DateTuple:
            enddatetime = cls.build_datetime(end, TupleBuilder.build_time())
            if enddatetime - datetime.datetime.min < durationobject:
                raise YearOutOfBoundsError('Interval end less than minimium date.')
        else:
            mindatetime = datetime.datetime.min
            if end.time.tz is not None:
                mindatetime = mindatetime.replace(tzinfo=endobject.tzinfo)
            if endobject - mindatetime < durationobject:
                raise YearOutOfBoundsError('Interval end less than minimium date.')
    else:
        startobject = cls._build_object(start)
        if type(start) is DateTuple:
            startdatetime = cls.build_datetime(start, TupleBuilder.build_time())
            if datetime.datetime.max - startdatetime < durationobject:
                raise YearOutOfBoundsError('Interval end greater than maximum date.')
        else:
            maxdatetime = datetime.datetime.max
            if start.time.tz is not None:
                maxdatetime = maxdatetime.replace(tzinfo=startobject.tzinfo)
            if maxdatetime - startobject < durationobject:
                raise YearOutOfBoundsError('Interval end greater than maximum date.')
    return (start, end, duration)