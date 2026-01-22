import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_filter_datastores_by_hubs(self):
    ds_values = []
    datastores = []
    for i in range(0, 10):
        value = 'ds-%d' % i
        ds_values.append(value)
        datastores.append(self._create_datastore(value))
    hubs = []
    hub_ids = ds_values[0:int(len(ds_values) / 2)]
    for hub_id in hub_ids:
        hub = mock.Mock()
        hub.hubId = hub_id
        hubs.append(hub)
    filtered_ds = pbm.filter_datastores_by_hubs(hubs, datastores)
    self.assertEqual(len(hubs), len(filtered_ds))
    filtered_ds_values = [vim_util.get_moref_value(ds) for ds in filtered_ds]
    self.assertEqual(set(hub_ids), set(filtered_ds_values))