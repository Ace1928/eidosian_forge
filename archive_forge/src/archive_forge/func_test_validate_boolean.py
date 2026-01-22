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
def test_validate_boolean(self):
    msg = validators.validate_boolean(True)
    self.assertIsNone(msg)
    msg = validators.validate_boolean(0)
    self.assertIsNone(msg)
    msg = validators.validate_boolean('false')
    self.assertIsNone(msg)
    msg = validators.validate_boolean('fasle')
    self.assertEqual("'fasle' is not a valid boolean value", msg)