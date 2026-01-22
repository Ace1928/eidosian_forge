from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def security_group_rule_delete(self, security_group_rule_id=None):
    """Delete a security group rule

        https://docs.openstack.org/api-ref/compute/#delete-security-group-rule

        :param string security_group_rule_id:
            Security group rule ID
        """
    url = '/os-security-group-rules'
    if security_group_rule_id is not None:
        return self.delete('/%s/%s' % (url, security_group_rule_id))
    return None