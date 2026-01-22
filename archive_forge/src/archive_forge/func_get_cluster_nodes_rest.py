from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_nodes_rest(self):
    """ get cluster node names, but the cluster may not exist yet
            return:
                None if the cluster cannot be reached
                a list of nodes
        """
    if self.node_records is None:
        records, error = rest_generic.get_0_or_more_records(self.rest_api, 'cluster/nodes', fields='name,uuid,cluster_interfaces')
        if error:
            self.module.fail_json(msg='Error fetching cluster node info: %s' % to_native(error), exception=traceback.format_exc())
        self.node_records = records or []
    return self.node_records