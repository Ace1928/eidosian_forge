import copy
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_types
def test_get_keys_from_resource_data(self):
    manager = mock.Mock()
    manager.api.client.get = mock.Mock(return_value=(200, {}))
    valid_extra_specs = {'test': 'test'}
    share_type = share_types.ShareType(mock.Mock(), {'extra_specs': valid_extra_specs, 'name': 'test'}, loaded=True)
    actual_result = share_type.get_keys()
    self.assertEqual(actual_result, valid_extra_specs)
    self.assertEqual(manager.api.client.get.call_count, 0)