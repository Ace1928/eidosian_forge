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
def get_quota_status(self):
    """
        Return details about the quota status
        :param:
            name : volume name
        :return: status of the quota. None if not found.
        :rtype: dict
        """
    quota_status_get = netapp_utils.zapi.NaElement('quota-status')
    quota_status_get.translate_struct({'volume': self.parameters['volume']})
    try:
        result = self.server.invoke_successfully(quota_status_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching quotas status info: %s' % to_native(error), exception=traceback.format_exc())
    return result['status']