from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_get_entity_no_uuid_like(self):
    self.cloud.use_direct_get = True
    name = 'name_no_uuid'
    resource = 'network'
    func = 'search_%ss' % resource
    filters = {}
    with mock.patch.object(self.cloud, func) as search:
        _utils._get_entity(self.cloud, resource, name, filters)
        search.assert_called_once_with(name, filters)