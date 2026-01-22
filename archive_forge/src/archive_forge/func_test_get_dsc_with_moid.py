from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_dsc_with_moid(self):
    session = mock.Mock()
    session.invoke_api = mock.Mock()
    session.invoke_api.return_value = 'ds-cluster'
    dsc_moid = 'group-p123'
    dsc_ref, dsc_name = datastore.get_dsc_ref_and_name(session, dsc_moid)
    self.assertEqual((dsc_moid, 'StoragePod'), (vim_util.get_moref_value(dsc_ref), vim_util.get_moref_type(dsc_ref)))
    self.assertEqual('ds-cluster', dsc_name)
    session.invoke_api.assert_called_once_with(vim_util, 'get_object_property', session.vim, mock.ANY, 'name')