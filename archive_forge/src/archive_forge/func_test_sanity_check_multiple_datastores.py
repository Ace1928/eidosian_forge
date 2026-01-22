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
def test_sanity_check_multiple_datastores(self):
    self.store.conf.glance_store.vmware_api_retry_count = 1
    self.store.conf.glance_store.vmware_task_poll_interval = 1
    self.store.conf.glance_store.vmware_datastores = ['a:b:0', 'a:d:0']
    try:
        self.store._sanity_check()
    except exceptions.BadStoreConfiguration:
        self.fail()