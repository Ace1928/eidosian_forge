import logging
from io import StringIO
import pytest
from ..batteryrunners import BatteryRunner, Report
def chk_warn(obj, fix=False):
    rep = Report(KeyError)
    if not 'anotherkey' in obj:
        rep.problem_level = 30
        rep.problem_msg = 'no "anotherkey"'
        if fix:
            obj['anotherkey'] = 'a string'
            rep.fix_msg = 'added "anotherkey"'
    return (obj, rep)