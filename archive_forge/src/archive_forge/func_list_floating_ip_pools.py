import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def list_floating_ip_pools(self):
    """List all available floating IP pools.

        NOTE: This function supports the nova-net view of the world. nova-net
        has been deprecated, so it's highly recommended to switch to using
        neutron. `get_external_ipv4_floating_networks` is what you should
        almost certainly be using.

        :returns: A list of floating IP pools
        """
    if not self._has_nova_extension('os-floating-ip-pools'):
        raise exc.OpenStackCloudUnavailableExtension('Floating IP pools extension is not available on target cloud')
    data = proxy._json_response(self.compute.get('os-floating-ip-pools'), error_message='Error fetching floating IP pool list')
    pools = self._get_and_munchify('floating_ip_pools', data)
    return [{'name': p['name']} for p in pools]