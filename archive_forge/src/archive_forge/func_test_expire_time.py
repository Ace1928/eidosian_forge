from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import pymacaroons
import pyrfc3339
from pymacaroons import Macaroon
def test_expire_time(self):
    ExpireTest = namedtuple('ExpireTest', 'about caveats expectTime')
    tests = [ExpireTest(about='no caveats', caveats=[], expectTime=None), ExpireTest(about='single time-before caveat', caveats=[fpcaveat(checkers.time_before_caveat(t1).condition)], expectTime=t1), ExpireTest(about='multiple time-before caveat', caveats=[fpcaveat(checkers.time_before_caveat(t2).condition), fpcaveat(checkers.time_before_caveat(t1).condition)], expectTime=t1), ExpireTest(about='mixed caveats', caveats=[fpcaveat(checkers.time_before_caveat(t1).condition), fpcaveat('allow bar'), fpcaveat(checkers.time_before_caveat(t2).condition), fpcaveat('deny foo')], expectTime=t1), ExpireTest(about='mixed caveats', caveats=[fpcaveat(checkers.COND_TIME_BEFORE + ' tomorrow')], expectTime=None)]
    for test in tests:
        print('test ', test.about)
        t = checkers.expiry_time(checkers.Namespace(), test.caveats)
        self.assertEqual(t, test.expectTime)