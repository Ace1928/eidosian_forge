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
def test_volume_reset_volume_status(self):
    self._verify('openstack.block_storage.v2.volume.Volume.reset_status', self.proxy.reset_volume_status, method_args=['value', '1', '2', '3'], expected_args=[self.proxy, '1', '2', '3'])