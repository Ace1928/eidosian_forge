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
def test_validate_range(self):
    msg = validators.validate_range(1, [1, 9])
    self.assertIsNone(msg)
    msg = validators.validate_range(5, [1, 9])
    self.assertIsNone(msg)
    msg = validators.validate_range(9, [1, 9])
    self.assertIsNone(msg)
    msg = validators.validate_range(1, (1, 9))
    self.assertIsNone(msg)
    msg = validators.validate_range(5, (1, 9))
    self.assertIsNone(msg)
    msg = validators.validate_range(9, (1, 9))
    self.assertIsNone(msg)
    msg = validators.validate_range(0, [1, 9])
    self.assertEqual("'0' is too small - must be at least '1'", msg)
    msg = validators.validate_range(10, (1, 9))
    self.assertEqual("'10' is too large - must be no larger than '9'", msg)
    msg = validators.validate_range('bogus', (1, 9))
    self.assertEqual("'bogus' is not an integer", msg)
    msg = validators.validate_range(10, (validators.UNLIMITED, validators.UNLIMITED))
    self.assertIsNone(msg)
    msg = validators.validate_range(10, (1, validators.UNLIMITED))
    self.assertIsNone(msg)
    msg = validators.validate_range(1, (validators.UNLIMITED, 9))
    self.assertIsNone(msg)
    msg = validators.validate_range(-1, (0, validators.UNLIMITED))
    self.assertEqual("'-1' is too small - must be at least '0'", msg)
    msg = validators.validate_range(10, (validators.UNLIMITED, 9))
    self.assertEqual("'10' is too large - must be no larger than '9'", msg)