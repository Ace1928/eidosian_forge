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
def test_validate_non_negative(self):
    msg = validators.validate_non_negative('abc')
    self.assertEqual("'abc' is not an integer", msg)
    for value in (-1, '-2'):
        self.assertEqual("'%s' should be non-negative" % value, validators.validate_non_negative(value))
    for value in (0, 1, '2', True, False):
        msg = validators.validate_non_negative(value)
        self.assertIsNone(msg)