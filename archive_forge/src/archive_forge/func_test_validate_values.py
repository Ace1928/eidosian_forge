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
def test_validate_values(self):
    msg = validators.validate_values(4)
    self.assertIsNone(msg)
    msg = validators.validate_values(4, [4, 6])
    self.assertIsNone(msg)
    msg = validators.validate_values(4, (4, 6))
    self.assertIsNone(msg)
    msg = validators.validate_values('1', ['2', '1', '4', '5'])
    self.assertIsNone(msg)
    response = "'valid_values' does not support membership operations"
    self.assertRaisesRegex(TypeError, response, validators.validate_values, data=None, valid_values=True)