from unittest import mock
from openstack.block_storage.v2 import _proxy
from openstack.block_storage.v2 import backup
from openstack.block_storage.v2 import capabilities
from openstack.block_storage.v2 import limits
from openstack.block_storage.v2 import quota_set
from openstack.block_storage.v2 import snapshot
from openstack.block_storage.v2 import stats
from openstack.block_storage.v2 import type
from openstack.block_storage.v2 import volume
from openstack import resource
from openstack.tests.unit import test_proxy_base
def test_set_snapshot_metadata(self):
    kwargs = {'a': '1', 'b': '2'}
    id = 'an_id'
    self._verify('openstack.block_storage.v2.snapshot.Snapshot.set_metadata', self.proxy.set_snapshot_metadata, method_args=[id], method_kwargs=kwargs, method_result=snapshot.Snapshot.existing(id=id, metadata=kwargs), expected_args=[self.proxy], expected_kwargs={'metadata': kwargs}, expected_result=snapshot.Snapshot.existing(id=id, metadata=kwargs))