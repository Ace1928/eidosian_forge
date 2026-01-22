from __future__ import annotations
from os import environ
from datetime import datetime, timedelta
from time import mktime as mktime_real
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.trial.unittest import SkipTest, TestCase
def setTZ(name: str | None) -> None:
    """
    Set time zone.

    @param name: a time zone name
    @type name: L{str}
    """
    if tzset is None:
        return
    if name is None:
        try:
            del environ['TZ']
        except KeyError:
            pass
    else:
        environ['TZ'] = name
    tzset()