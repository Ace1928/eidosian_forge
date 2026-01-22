from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_list(self, limit=None, marker=None, search_opts=None):
    """Get security groups

        https://docs.openstack.org/api-ref/compute/#list-security-groups

        :param integer limit:
            query return count limit
        :param string marker:
            query marker
        :param search_opts:
            (undocumented) Search filter dict
            all_tenants: True|False - return all projects
        :returns:
            list of security groups names
        """
    params = {}
    if search_opts is not None:
        params = dict(((k, v) for k, v in search_opts.items() if v))
    if limit:
        params['limit'] = limit
    if marker:
        params['offset'] = marker
    url = '/os-security-groups'
    return self.list(url, **params)['security_groups']