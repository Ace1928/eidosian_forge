from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_set(self, security_group=None, **params):
    """Update a security group

        https://docs.openstack.org/api-ref/compute/#update-security-group

        :param string security_group:
            Security group name or ID

        TODO(dtroyer): Create an update method in osc-lib
        """
    if params is None:
        return None
    url = '/os-security-groups'
    security_group = self.find(url, attr='name', value=security_group)
    if security_group is not None:
        for k, v in params.items():
            if k in security_group:
                security_group[k] = v
        return self._request('PUT', '/%s/%s' % (url, security_group['id']), json={'security_group': security_group}).json()['security_group']
    return None