import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_convert_datastores_to_hubs(self):
    ds_values = []
    datastores = []
    for i in range(0, 10):
        value = 'ds-%d' % i
        ds_values.append(value)
        datastores.append(self._create_datastore(value))
    pbm_client_factory = mock.Mock()
    pbm_client_factory.create.side_effect = lambda *args: mock.Mock()
    hubs = pbm.convert_datastores_to_hubs(pbm_client_factory, datastores)
    self.assertEqual(len(datastores), len(hubs))
    hub_ids = [hub.hubId for hub in hubs]
    self.assertEqual(set(ds_values), set(hub_ids))