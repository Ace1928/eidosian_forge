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
def test_validate_list_of_unique_strings(self):
    data = 'TEST'
    msg = validators.validate_list_of_unique_strings(data, None)
    self.assertEqual("'TEST' is not a list", msg)
    data = ['TEST01', 'TEST02', 'TEST01']
    msg = validators.validate_list_of_unique_strings(data, None)
    self.assertEqual("Duplicate items in the list: 'TEST01'", msg)
    data = ['12345678', '123456789']
    msg = validators.validate_list_of_unique_strings(data, 8)
    self.assertEqual("'123456789' exceeds maximum length of 8", msg)
    data = ['TEST01', 'TEST02', 'TEST03']
    msg = validators.validate_list_of_unique_strings(data, None)
    self.assertIsNone(msg)