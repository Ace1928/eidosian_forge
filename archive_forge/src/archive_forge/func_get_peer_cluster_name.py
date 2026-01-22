from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_peer_cluster_name(self):
    """
        Get local cluster name
        :return: cluster name
        """
    if self.use_rest:
        return self.get_peer_cluster_name_rest()
    cluster_info = netapp_utils.zapi.NaElement('cluster-identity-get')
    server = self.dest_server if self.is_remote_peer() else self.server
    try:
        result = server.invoke_successfully(cluster_info, enable_tunneling=True)
        return result.get_child_by_name('attributes').get_child_by_name('cluster-identity-info').get_child_content('cluster-name')
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching peer cluster name for peer vserver %s: %s' % (self.parameters['peer_vserver'], to_native(error)), exception=traceback.format_exc())