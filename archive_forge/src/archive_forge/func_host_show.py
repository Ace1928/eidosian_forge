from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def host_show(self, host=None):
    """Show host

        https://docs.openstack.org/api-ref/compute/#show-host-details
        Valid for Compute 2.0 - 2.42
        """
    url = '/os-hosts'
    r_host = self.find(url, attr='host_name', value=host)
    data = []
    for h in r_host:
        data.append(h['resource'])
    return data