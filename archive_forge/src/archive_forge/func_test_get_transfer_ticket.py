from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_transfer_ticket(self):
    dc_path = 'datacenter-1'
    ds_name = 'datastore-1'
    params = {'dcPath': dc_path, 'dsName': ds_name}
    query = urlparse.urlencode(params)
    url = 'https://13.37.73.31/folder/images/aa.vmdk?%s' % query
    session = mock.Mock()
    session.invoke_api = mock.Mock()

    class Ticket(object):
        id = 'fake_id'
    session.invoke_api.return_value = Ticket()
    ds_url = datastore.DatastoreURL.urlparse(url)
    ticket = ds_url.get_transfer_ticket(session, 'PUT')
    self.assertEqual('%s="%s"' % (constants.CGI_COOKIE_KEY, 'fake_id'), ticket)