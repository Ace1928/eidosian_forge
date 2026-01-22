from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_datastore_by_ref(self):
    session = mock.Mock()
    ds_ref = mock.Mock()
    expected_props = {'summary.name': 'datastore1', 'summary.type': 'NFS', 'summary.freeSpace': 1000, 'summary.capacity': 2000}
    session.invoke_api = mock.Mock()
    session.invoke_api.return_value = expected_props
    ds_obj = datastore.get_datastore_by_ref(session, ds_ref)
    self.assertEqual(expected_props['summary.name'], ds_obj.name)
    self.assertEqual(expected_props['summary.type'], ds_obj.type)
    self.assertEqual(expected_props['summary.freeSpace'], ds_obj.freespace)
    self.assertEqual(expected_props['summary.capacity'], ds_obj.capacity)