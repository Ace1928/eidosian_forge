import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_validate_ip_address_with_leading_zero(self):
    ip_addr = '1.1.1.01'
    expected_msg = "'%(data)s' is not an accepted IP address, '%(ip)s' is recommended"
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual(expected_msg % {'data': ip_addr, 'ip': '1.1.1.1'}, msg)
    ip_addr = '1.1.1.011'
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual(expected_msg % {'data': ip_addr, 'ip': '1.1.1.11'}, msg)
    ip_addr = '1.1.1.09'
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual(expected_msg % {'data': ip_addr, 'ip': '1.1.1.9'}, msg)
    ip_addr = 'fe80:0:0:0:0:0:0:0001'
    msg = validators.validate_ip_address(ip_addr)
    self.assertIsNone(msg)