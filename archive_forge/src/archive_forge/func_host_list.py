from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def host_list(self, zone=None):
    """Lists hypervisor Hosts

        https://docs.openstack.org/api-ref/compute/#list-hosts
        Valid for Compute 2.0 - 2.42

        :param string zone:
            Availability zone
        :returns: A dict of the floating IP attributes
        """
    url = '/os-hosts'
    if zone:
        url = '/os-hosts?zone=%s' % zone
    return self.list(url)['hosts']