from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def destroy_fcp(self):
    """
        Destroys an already stopped FCP
        :return:
        """
    try:
        self.server.invoke_successfully(netapp_utils.zapi.NaElement('fcp-service-destroy'), True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error destroying FCP %s' % to_native(error), exception=traceback.format_exc())