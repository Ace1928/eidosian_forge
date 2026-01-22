from unittest import mock
from keystoneauth1 import adapter
from openstack.common import metadata
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_set_metadata_item(self):
    res = self.sot
    res.metadata = {'foo': 'bar'}
    result = res.set_metadata_item(self.session, 'foo', 'black')
    self.assertEqual({'foo': 'black'}, res.metadata)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/metadata/foo'
    self.session.put.assert_called_once_with(url, json={'meta': {'foo': 'black'}})