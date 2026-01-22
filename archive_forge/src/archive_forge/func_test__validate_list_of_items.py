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
def test__validate_list_of_items(self):
    items = [None, 123, 'e5069610-744b-42a7-8bd8-ceac1a229cd4', '12345678123456781234567812345678', {'uuid': 'e5069610-744b-42a7-8bd8-ceac1a229cd4'}]
    for item in items:
        msg = validators._validate_list_of_items(mock.Mock(), item)
        error = "'%s' is not a list" % item
        self.assertEqual(error, msg)
    duplicate_items = ['e5069610-744b-42a7-8bd8-ceac1a229cd4', 'f3eeab00-8367-4524-b662-55e64d4cacb5', 'e5069610-744b-42a7-8bd8-ceac1a229cd4']
    msg = validators._validate_list_of_items(mock.Mock(), duplicate_items)
    error = "Duplicate items in the list: 'e5069610-744b-42a7-8bd8-ceac1a229cd4'"
    self.assertEqual(error, msg)
    valid_lists = [[], [1, 2, 3], ['a', 'b', 'c']]
    for list_obj in valid_lists:
        msg = validators._validate_list_of_items(mock.Mock(return_value=None), list_obj)
        self.assertIsNone(msg)