from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_server_and_peer_address(self, cluster):
    if cluster == 'source':
        server = self.rest_api if self.use_rest else self.server
        return (server, self.parameters['dest_intercluster_lifs'])
    server = self.dst_rest_api if self.use_rest else self.dest_server
    return (server, self.parameters['source_intercluster_lifs'])