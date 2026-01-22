from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cluster_rest(self, older_api=False):
    """
        Create a cluster
        """
    query = None
    body = self.create_cluster_body(nodes=self.create_nodes())
    if 'single_node_cluster' in body:
        query = {'single_node_cluster': body.pop('single_node_cluster')}
    dummy, error = rest_generic.post_async(self.rest_api, 'cluster', body, query, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error creating cluster %s: %s' % (self.parameters['cluster_name'], to_native(error)), exception=traceback.format_exc())