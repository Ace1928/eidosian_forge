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
def fake_datastore_obj(*args, **kwargs):
    dc_obj = oslo_datacenter.Datacenter(ref='fake-ref', name='fake-name')
    dc_obj.path = args[0]
    return oslo_datastore.Datastore(ref='fake-ref', datacenter=dc_obj, name=args[1])