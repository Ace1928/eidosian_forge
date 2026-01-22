from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def on_or_off_quota(self, status, cd_action=None):
    """
        on or off quota
        """
    quota = netapp_utils.zapi.NaElement.create_node_with_children(status, **{'volume': self.parameters['volume']})
    try:
        self.server.invoke_successfully(quota, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if cd_action == 'delete' and status == 'quota-on' and ('14958:No valid quota rules found' in to_native(error)):
            self.module.warn('Last rule deleted, quota is off.')
            return
        self.module.fail_json(msg='Error setting %s for %s: %s' % (status, self.parameters['volume'], to_native(error)), exception=traceback.format_exc())