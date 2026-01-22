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
@mock.patch('oslo_vmware.api.VMwareAPISession')
def test_http_get_redirect(self, mock_api_session):
    redirect1 = {'location': 'https://example.com?dsName=ds1&dcPath=dc1'}
    redirect2 = {'location': 'https://example.com?dsName=ds2&dcPath=dc2'}
    responses = [utils.fake_response(), utils.fake_response(status_code=302, headers=redirect1), utils.fake_response(status_code=301, headers=redirect2)]

    def getresponse(*args, **kwargs):
        return responses.pop()
    expected_image_size = 31
    expected_returns = ['I am a teapot, short and stout\n']
    loc = location.get_location_from_uri('vsphere://127.0.0.1/folder/openstack_glance/%s?dsName=ds1&dcPath=dc1' % FAKE_UUID, conf=self.conf)
    with mock.patch('requests.Session.request') as HttpConn:
        HttpConn.side_effect = getresponse
        image_file, image_size = self.store.get(loc)
    self.assertEqual(expected_image_size, image_size)
    chunks = [c for c in image_file]
    self.assertEqual(expected_returns, chunks)