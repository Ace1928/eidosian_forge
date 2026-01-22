from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_nvme(self, status=None):
    """
        Modify NVMe service
        """
    if status is None:
        status = self.parameters['status_admin']
    if self.use_rest:
        return self.modify_nvme_rest(status)
    options = {'is-available': status}
    nvme_modify = netapp_utils.zapi.NaElement('nvme-modify')
    nvme_modify.translate_struct(options)
    try:
        self.server.invoke_successfully(nvme_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying nvme for vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())