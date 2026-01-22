from datetime import (
from typing import Optional
@classmethod
def fromSignHoursMinutes(cls, sign: str, hours: int, minutes: int) -> 'FixedOffsetTimeZone':
    """
        Construct a L{FixedOffsetTimeZone} from an offset described by sign
        ('+' or '-'), hours, and minutes.

        @note: For protocol compatibility with AMP, this method never uses 'Z'

        @param sign: A string describing the positive or negative-ness of the
            offset.
        @param hours: The number of hours in the offset.
        @param minutes: The number of minutes in the offset

        @return: A time zone with the given offset, and a name describing the
            offset.
        """
    name = '%s%02i:%02i' % (sign, hours, minutes)
    if sign == '-':
        hours = -hours
        minutes = -minutes
    elif sign != '+':
        raise ValueError(f'Invalid sign for timezone {sign!r}')
    return cls(TimeDelta(hours=hours, minutes=minutes), name)