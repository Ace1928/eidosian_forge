from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
@mock.patch('oslo_config.cfg.CONF')
def test_convert_none_to_mac_address(self, CONF):
    CONF.base_mac = 'fa:16:3e:00:00:00'
    self.assertTrue(netaddr.valid_mac(converters.convert_to_mac_if_none(None)))