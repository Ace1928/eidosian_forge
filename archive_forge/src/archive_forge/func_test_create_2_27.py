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
@ddt.data(*get_valid_type_create_data_2_27())
@ddt.unpack
def test_create_2_27(self, is_public, dhss, snapshot, create_from_snapshot, revert_to_snapshot, extra_specs):
    extra_specs = copy.copy(extra_specs)
    extra_specs = self._add_standard_extra_specs_to_dict(extra_specs, create_from_snapshot=create_from_snapshot, revert_to_snapshot=revert_to_snapshot)
    manager = self._get_share_types_manager('2.27')
    self.mock_object(manager, '_create', mock.Mock(return_value='fake'))
    result = manager.create('test-type-3', spec_driver_handles_share_servers=dhss, spec_snapshot_support=snapshot, extra_specs=extra_specs, is_public=is_public)
    expected_extra_specs = dict(extra_specs or {})
    expected_extra_specs['driver_handles_share_servers'] = dhss
    if snapshot is None:
        expected_extra_specs.pop('snapshot_support', None)
    else:
        expected_extra_specs['snapshot_support'] = snapshot
    if create_from_snapshot is None:
        expected_extra_specs.pop('create_share_from_snapshot_support', None)
    else:
        expected_extra_specs['create_share_from_snapshot_support'] = create_from_snapshot
    if revert_to_snapshot is None:
        expected_extra_specs.pop('revert_to_snapshot_support', None)
    else:
        expected_extra_specs['revert_to_snapshot_support'] = revert_to_snapshot
    expected_body = {'share_type': {'name': 'test-type-3', 'share_type_access:is_public': is_public, 'extra_specs': expected_extra_specs}}
    manager._create.assert_called_once_with('/types', expected_body, 'share_type')
    self.assertEqual('fake', result)