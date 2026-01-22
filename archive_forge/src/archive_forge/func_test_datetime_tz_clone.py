import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_datetime_tz_clone(self):
    now = datetime.datetime.now()
    if zoneinfo:
        eastern = zoneinfo.ZoneInfo('US/Eastern')
        e_dt = now.replace(tzinfo=eastern)
    else:
        eastern = timezone('US/Eastern')
        e_dt = eastern.localize(now)
    e_dt2 = _dumps_loads(e_dt)
    self.assertEqual(e_dt, e_dt2)
    self.assertEqual(e_dt.strftime(_TZ_FMT), e_dt2.strftime(_TZ_FMT))