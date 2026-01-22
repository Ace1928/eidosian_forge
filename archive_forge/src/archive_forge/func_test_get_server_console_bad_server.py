import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_get_server_console_bad_server(self):
    self.assertRaises(exceptions.SDKException, self.user_cloud.get_server_console, server=self.server_name)