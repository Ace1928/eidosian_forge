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
def test_validate_string(self):
    msg = validators.validate_string(None, None)
    self.assertEqual("'None' is not a valid string", msg)
    msg = validators.validate_string('', 0)
    self.assertIsNone(msg)
    msg = validators.validate_string('', 9)
    self.assertIsNone(msg)
    msg = validators.validate_string('123456789', 10)
    self.assertIsNone(msg)
    msg = validators.validate_string('123456789', 9)
    self.assertIsNone(msg)
    msg = validators.validate_string('1234567890', 9)
    self.assertEqual("'1234567890' exceeds maximum length of 9", msg)
    msg = validators.validate_string('123456789', None)
    self.assertIsNone(msg)