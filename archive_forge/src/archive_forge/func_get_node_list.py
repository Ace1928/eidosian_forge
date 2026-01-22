from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def get_node_list(self):
    """
            Get Node List
            :description: Find and retrieve node_ids from the active cluster

            :return: None
            :rtype: None
        """
    action_nodes_list = list()
    if len(self.node_ids) > 0:
        unprocessed_node_list = list(self.node_ids)
        list_nodes = []
        try:
            all_nodes = self.sfe.list_all_nodes()
        except netapp_utils.solidfire.common.ApiServerError as exception_object:
            self.module.fail_json(msg='Error getting list of nodes from cluster: %s' % to_native(exception_object), exception=traceback.format_exc())
        if self.state == 'present':
            list_nodes = all_nodes.pending_nodes
        else:
            list_nodes = all_nodes.nodes
        for current_node in list_nodes:
            if self.state == 'absent' and (str(current_node.node_id) in self.node_ids or current_node.name in self.node_ids or current_node.mip in self.node_ids):
                if self.check_node_has_active_drives(current_node.node_id):
                    self.module.fail_json(msg='Error deleting node %s: node has active drives' % current_node.name)
                else:
                    action_nodes_list.append(current_node.node_id)
            if self.state == 'present' and (str(current_node.pending_node_id) in self.node_ids or current_node.name in self.node_ids or current_node.mip in self.node_ids):
                action_nodes_list.append(current_node.pending_node_id)
        if self.state == 'present':
            for current_node in all_nodes.nodes:
                if str(current_node.node_id) in unprocessed_node_list:
                    unprocessed_node_list.remove(str(current_node.node_id))
                elif current_node.name in unprocessed_node_list:
                    unprocessed_node_list.remove(current_node.name)
                elif current_node.mip in unprocessed_node_list:
                    unprocessed_node_list.remove(current_node.mip)
            for current_node in all_nodes.pending_nodes:
                if str(current_node.pending_node_id) in unprocessed_node_list:
                    unprocessed_node_list.remove(str(current_node.pending_node_id))
                elif current_node.name in unprocessed_node_list:
                    unprocessed_node_list.remove(current_node.name)
                elif current_node.mip in unprocessed_node_list:
                    unprocessed_node_list.remove(current_node.mip)
            if len(unprocessed_node_list) > 0:
                summary = dict(nodes=self.extract_node_info(all_nodes.nodes), pending_nodes=self.extract_node_info(all_nodes.pending_nodes), pending_active_nodes=self.extract_node_info(all_nodes.pending_active_nodes))
                self.module.fail_json(msg='Error adding nodes %s: nodes not in pending or active lists: %s' % (to_native(unprocessed_node_list), repr(summary)))
    return action_nodes_list