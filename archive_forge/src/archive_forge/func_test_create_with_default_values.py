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
@ddt.data(('2.6', True), ('2.7', True), ('2.24', True), ('2.41', True), ('2.6', False), ('2.7', False), ('2.24', False), ('2.41', False))
@ddt.unpack
def test_create_with_default_values(self, microversion, dhss):
    manager = self._get_share_types_manager(microversion)
    self.mock_object(manager, '_create', mock.Mock(return_value='fake'))
    description = 'test description'
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.41'):
        result = manager.create('test-type-3', dhss, description=description)
    else:
        result = manager.create('test-type-3', dhss)
    if api_versions.APIVersion(microversion) > api_versions.APIVersion('2.6'):
        is_public_keyname = 'share_type_access:is_public'
    else:
        is_public_keyname = 'os-share-type-access:is_public'
    expected_body = {'share_type': {'name': 'test-type-3', is_public_keyname: True, 'extra_specs': {'driver_handles_share_servers': dhss, 'snapshot_support': True}}}
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.24'):
        del expected_body['share_type']['extra_specs']['snapshot_support']
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.41'):
        expected_body['share_type']['description'] = description
    manager._create.assert_called_once_with('/types', expected_body, 'share_type')
    self.assertEqual('fake', result)