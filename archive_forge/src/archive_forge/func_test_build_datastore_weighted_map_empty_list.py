import hashlib
import io
from unittest import mock
import uuid
from oslo_utils import secretutils
from oslo_utils import units
from oslo_vmware import api
from oslo_vmware import exceptions as vmware_exceptions
from oslo_vmware.objects import datacenter as oslo_datacenter
from oslo_vmware.objects import datastore as oslo_datastore
import glance_store._drivers.vmware_datastore as vm_store
from glance_store import backend
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
@mock.patch.object(vm_store.Store, '_get_datastore')
@mock.patch.object(api, 'VMwareAPISession')
def test_build_datastore_weighted_map_empty_list(self, mock_api_session, mock_ds_ref):
    datastores = []
    ret = self.store._build_datastore_weighted_map(datastores)
    self.assertEqual({}, ret)