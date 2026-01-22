from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_delete(self, security_group=None):
    """Delete a security group

        https://docs.openstack.org/api-ref/compute/#delete-security-group

        :param string security_group:
            Security group name or ID
        """
    url = '/os-security-groups'
    security_group = self.find(url, attr='name', value=security_group)['id']
    if security_group is not None:
        return self.delete('/%s/%s' % (url, security_group))
    return None