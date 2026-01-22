from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_peer_modify_rest(self, cluster, uuid, modified_peer_addresses):
    api = 'cluster/peers'
    body = {'remote.ip_addresses': modified_peer_addresses}
    server = self.rest_api if cluster == 'source' else self.dst_rest_api
    dummy, error = rest_generic.patch_async(server, api, uuid, body)
    if error:
        self.module.fail_json(msg=error)