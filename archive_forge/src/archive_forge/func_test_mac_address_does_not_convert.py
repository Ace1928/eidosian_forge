from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_mac_address_does_not_convert(self):
    valid_mac = 'fa:16:3e:b6:78:1f'
    self.assertEqual(valid_mac, converters.convert_to_mac_if_none(valid_mac))