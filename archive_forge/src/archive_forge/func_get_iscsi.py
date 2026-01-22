from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_iscsi(self):
    """
        Return details about the iscsi service

        :return: Details about the iscsi service
        :rtype: dict
        """
    if self.use_rest:
        return self.get_iscsi_rest()
    iscsi_info = netapp_utils.zapi.NaElement('iscsi-service-get-iter')
    iscsi_attributes = netapp_utils.zapi.NaElement('iscsi-service-info')
    iscsi_attributes.add_new_child('vserver', self.parameters['vserver'])
    query = netapp_utils.zapi.NaElement('query')
    query.add_child_elem(iscsi_attributes)
    iscsi_info.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(iscsi_info, True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error finding iscsi service in %s: %s' % (self.parameters['vserver'], to_native(e)), exception=traceback.format_exc())
    return_value = None
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        iscsi = result.get_child_by_name('attributes-list').get_child_by_name('iscsi-service-info')
        if iscsi:
            is_started = 'started' if iscsi.get_child_content('is-available') == 'true' else 'stopped'
            return_value = {'service_state': is_started}
    return return_value