from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_node_action(self):
    node_action = None
    if self.parameters.get('cluster_ip_address') is not None:
        existing_interfaces = self.get_cluster_ip_address(self.parameters.get('cluster_ip_address'))
        if self.parameters.get('state') == 'present':
            node_action = 'add_node' if existing_interfaces is None else None
        else:
            node_action = 'remove_node' if existing_interfaces is not None else None
    if self.parameters.get('node_name') is not None and self.parameters['state'] == 'absent':
        nodes = self.get_cluster_nodes()
        if self.parameters.get('node_name') in nodes:
            node_action = 'remove_node'
    if node_action is not None:
        self.na_helper.changed = True
    return node_action