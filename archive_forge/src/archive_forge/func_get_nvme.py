from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_nvme(self):
    """
        Get current nvme details
        :return: dict if nvme exists, None otherwise
        """
    if self.use_rest:
        return self.get_nvme_rest()
    nvme_get = netapp_utils.zapi.NaElement('nvme-get-iter')
    query = {'query': {'nvme-target-service-info': {'vserver': self.parameters['vserver']}}}
    nvme_get.translate_struct(query)
    try:
        result = self.server.invoke_successfully(nvme_get, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching nvme info: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attributes_list = result.get_child_by_name('attributes-list')
        nvme_info = attributes_list.get_child_by_name('nvme-target-service-info')
        return {'status_admin': self.na_helper.get_value_for_bool(True, nvme_info.get_child_content('is-available'))}
    return None