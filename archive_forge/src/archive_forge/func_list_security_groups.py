from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def list_security_groups(self, filters=None):
    """List all available security groups.

        :param filters: (optional) dict of filter conditions to push down
        :returns: A list of security group
            ``openstack.network.v2.security_group.SecurityGroup``.

        """
    if not self._has_secgroups():
        raise exc.OpenStackCloudUnavailableFeature('Unavailable feature: security groups')
    if not filters:
        filters = {}
    data = []
    if self._use_neutron_secgroups():
        return list(self.network.security_groups(**filters))
    else:
        data = proxy._json_response(self.compute.get('/os-security-groups', params=filters))
    return self._normalize_secgroups(self._get_and_munchify('security_groups', data))