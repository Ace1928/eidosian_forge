from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_motd_zapi(self):
    motd_get_iter = netapp_utils.zapi.NaElement('vserver-motd-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    motd_info = netapp_utils.zapi.NaElement('vserver-motd-info')
    motd_info.add_new_child('vserver', self.parameters['vserver'])
    query.add_child_elem(motd_info)
    motd_get_iter.add_child_elem(query)
    try:
        result = self.server.invoke_successfully(motd_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching motd info: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        motd_info = result.get_child_by_name('attributes-list').get_child_by_name('vserver-motd-info')
        motd_message = motd_info.get_child_content('message')
        motd_message = str(motd_message).rstrip()
        if motd_message == 'None':
            motd_message = ''
        show_cluster_motd = motd_info.get_child_content('is-cluster-message-enabled') == 'true'
        return (motd_message, show_cluster_motd)
    return ('', False)