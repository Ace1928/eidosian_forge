from unittest import mock
from openstack.block_storage.v3 import _proxy
from openstack.block_storage.v3 import backup
from openstack.block_storage.v3 import capabilities
from openstack.block_storage.v3 import extension
from openstack.block_storage.v3 import group
from openstack.block_storage.v3 import group_snapshot
from openstack.block_storage.v3 import group_type
from openstack.block_storage.v3 import limits
from openstack.block_storage.v3 import quota_set
from openstack.block_storage.v3 import resource_filter
from openstack.block_storage.v3 import service
from openstack.block_storage.v3 import snapshot
from openstack.block_storage.v3 import stats
from openstack.block_storage.v3 import type
from openstack.block_storage.v3 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
def test_manage_snapshot(self):
    kwargs = {'volume_id': 'fake_id', 'remote_source': 'fake_volume', 'snapshot_name': 'fake_snap', 'description': 'test_snap', 'property': {'k': 'v'}}
    self._verify('openstack.block_storage.v3.snapshot.Snapshot.manage', self.proxy.manage_snapshot, method_kwargs=kwargs, method_result=snapshot.Snapshot(id='fake_id'), expected_args=[self.proxy], expected_kwargs=kwargs, expected_result=snapshot.Snapshot(id='fake_id'))