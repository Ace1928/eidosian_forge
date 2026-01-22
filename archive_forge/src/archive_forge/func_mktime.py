from __future__ import annotations
from os import environ
from datetime import datetime, timedelta
from time import mktime as mktime_real
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.trial.unittest import SkipTest, TestCase
def mktime(t9: tuple[int, int, int, int, int, int, int, int, int]) -> float:
    """
    Call L{mktime_real}, and if it raises L{OverflowError}, catch it and raise
    SkipTest instead.

    @param t9: A time as a 9-item tuple.
    @type t9: L{tuple}

    @return: A timestamp.
    @rtype: L{float}
    """
    try:
        return mktime_real(t9)
    except OverflowError:
        raise SkipTest(f'Platform cannot construct time zone for {t9!r}')