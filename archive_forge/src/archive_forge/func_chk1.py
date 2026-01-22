import logging
from io import StringIO
import pytest
from ..batteryrunners import BatteryRunner, Report
def chk1(obj, fix=False):
    rep = Report(KeyError)
    if 'testkey' in obj:
        return (obj, rep)
    rep.problem_level = 20
    rep.problem_msg = 'no "testkey"'
    if fix:
        obj['testkey'] = 1
        rep.fix_msg = 'added "testkey"'
    return (obj, rep)