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
def test_validate_ipv4_invalid(self):
    testdata = '300.0.0.1'
    self.assertEqual("'300.0.0.1' is neither a valid IP address, nor is it a valid IP subnet", validators.validate_ip_or_subnet_or_none(testdata))