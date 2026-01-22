from unittest import mock
from keystoneauth1 import adapter
from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_get_metadata_item_not_exists(self):
    res = self.sot
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.content = None
    self.session.get.side_effect = [mock_response]
    self.assertRaises(exceptions.NotFoundException, res.get_metadata_item, self.session, 'dummy')