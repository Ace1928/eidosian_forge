from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_convert_ipv4_address_with_CIDR(self):
    result = converters.convert_cidr_to_canonical_format('192.168.1.1/24')
    self.assertEqual('192.168.1.1/24', result)