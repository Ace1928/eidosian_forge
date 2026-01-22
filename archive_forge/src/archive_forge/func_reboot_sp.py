from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def reboot_sp(self):
    if self.use_rest:
        return self.reboot_sp_rest()
    reboot = netapp_utils.zapi.NaElement('service-processor-reboot')
    reboot.add_new_child('node', self.parameters['node'])
    try:
        self.server.invoke_successfully(reboot, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error rebooting service processor: %s' % to_native(error), exception=traceback.format_exc())