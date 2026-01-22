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
def test_wrong_cidr(self):
    wrong_cidrs = ['10.500.5.0/24', '10.0.0.1/40', '10.0.0.10.0/24', 'cidr', '', '2001:db8:5000::/64', '2001:db8::/130']
    for cidr in wrong_cidrs:
        self.assertRaises(exception.DBError, self._add_row, id=uuidutils.generate_uuid(), cidr=cidr)