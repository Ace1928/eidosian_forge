from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_nodes(self, ignore_error=True):
    """ get cluster node names, but the cluster may not exist yet
            return:
                None if the cluster cannot be reached
                a list of nodes
        """
    if self.use_rest:
        return self.get_cluster_node_names_rest()
    zapi = netapp_utils.zapi.NaElement('cluster-node-get-iter')
    try:
        result = self.server.invoke_successfully(zapi, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if ignore_error:
            return None
        self.module.fail_json(msg='Error fetching cluster node info: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('attributes-list'):
        cluster_nodes = []
        for node_info in result.get_child_by_name('attributes-list').get_children():
            node_name = node_info.get_child_content('node-name')
            if node_name is not None:
                cluster_nodes.append(node_name)
        return cluster_nodes
    return None