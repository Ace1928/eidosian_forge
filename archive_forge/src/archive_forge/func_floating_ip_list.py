from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def floating_ip_list(self):
    """Get floating IPs

        https://docs.openstack.org/api-ref/compute/#show-floating-ip-address-details

        :returns:
            list of floating IPs
        """
    url = '/os-floating-ips'
    return self.list(url)['floating_ips']