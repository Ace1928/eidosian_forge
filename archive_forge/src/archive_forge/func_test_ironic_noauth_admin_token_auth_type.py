import openstack.cloud
from openstack.tests.unit import base
def test_ironic_noauth_admin_token_auth_type(self):
    """Test noauth selection for Ironic in OpenStackCloud

        The old way of doing this was to abuse admin_token.
        """
    self.cloud_noauth = openstack.connect(auth_type='admin_token', auth=dict(endpoint='https://baremetal.example.com/v1', token='ignored'))
    self.cloud_noauth.list_machines()
    self.assert_calls()