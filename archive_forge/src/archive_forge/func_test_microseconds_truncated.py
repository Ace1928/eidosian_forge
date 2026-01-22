import abc
import netaddr
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_utils import timeutils
from oslo_utils import uuidutils
import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import sqlalchemytypes
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_microseconds_truncated(self):
    tstamp = timeutils.utcnow()
    tstamp_low = tstamp.replace(microsecond=111111)
    tstamp_high = tstamp.replace(microsecond=999999)
    self._add_row(id=1, thetime=tstamp_low)
    self._add_row(id=2, thetime=tstamp_high)
    rows = self._get_all()
    self.assertEqual(2, len(rows))
    self.assertEqual(rows[0].thetime, rows[1].thetime)