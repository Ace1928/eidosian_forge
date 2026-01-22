from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_delete(self, cluster, uuid=None):
    """
        Delete a cluster peer on source or destination
        For source cluster, peer cluster-name = destination cluster name and vice-versa
        :param cluster: type of cluster (source or destination)
        :return:
        """
    if self.use_rest:
        return self.cluster_peer_delete_rest(cluster, uuid)
    if cluster == 'source':
        server, peer_cluster_name = (self.server, self.parameters['dest_cluster_name'])
    else:
        server, peer_cluster_name = (self.dest_server, self.parameters['source_cluster_name'])
    cluster_peer_delete = netapp_utils.zapi.NaElement.create_node_with_children('cluster-peer-delete', **{'cluster-name': peer_cluster_name})
    try:
        server.invoke_successfully(cluster_peer_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting cluster peer %s: %s' % (peer_cluster_name, to_native(error)), exception=traceback.format_exc())