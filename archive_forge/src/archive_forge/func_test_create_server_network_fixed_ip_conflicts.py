import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_network_fixed_ip_conflicts(self):
    """
        Verify that if 'fixed_ip' and 'v4-fixed-ip' are both supplied in nics,
        we throw an exception.
        """
    self.use_nothing()
    fixed_ip = '10.0.0.1'
    self.assertRaises(exceptions.SDKException, self.cloud.create_server, 'server-name', dict(id='image-id'), dict(id='flavor-id'), nics=[{'fixed_ip': fixed_ip, 'v4-fixed-ip': fixed_ip}])
    self.assert_calls()