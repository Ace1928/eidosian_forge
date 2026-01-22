import openstack.cloud
from openstack.tests.unit import base
def test_ironic_noauth_auth_endpoint(self):
    """Test noauth selection for Ironic in OpenStackCloud

        Sometimes people also write clouds.yaml files that look like this:

        ::
          clouds:
            bifrost:
              auth_type: "none"
              endpoint: https://baremetal.example.com
        """
    self.cloud_noauth = openstack.connect(auth_type='none', endpoint='https://baremetal.example.com/')
    self.cloud_noauth.list_machines()
    self.assert_calls()