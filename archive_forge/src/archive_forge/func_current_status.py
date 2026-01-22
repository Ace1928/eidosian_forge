from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def current_status(self):
    try:
        status = self.server.invoke_successfully(netapp_utils.zapi.NaElement('fcp-service-status'), True)
        return status.get_child_content('is-available') == 'true'
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error destroying FCP: %s' % to_native(error), exception=traceback.format_exc())