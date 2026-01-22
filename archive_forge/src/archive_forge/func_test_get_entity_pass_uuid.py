from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_get_entity_pass_uuid(self):
    uuid = uuid4().hex
    self.cloud.use_direct_get = True
    resources = ['flavor', 'image', 'volume', 'network', 'subnet', 'port', 'floating_ip', 'security_group']
    for r in resources:
        f = 'get_%s_by_id' % r
        with mock.patch.object(self.cloud, f) as get:
            _utils._get_entity(self.cloud, r, uuid, {})
            get.assert_called_once_with(uuid)