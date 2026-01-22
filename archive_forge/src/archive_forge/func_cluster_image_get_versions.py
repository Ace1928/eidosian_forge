from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_get_versions(self):
    """
        Get current cluster image versions for each node
        :return: list of tuples (node_id, node_version) or empty list
        """
    if self.use_rest:
        return self.cluster_image_get_rest('versions')
    cluster_image_get_iter = self.cluster_image_get_iter()
    try:
        result = self.server.invoke_successfully(cluster_image_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching cluster image details: %s: %s' % (self.parameters['package_version'], to_native(error)), exception=traceback.format_exc())
    return [(image_info.get_child_content('node-id'), image_info.get_child_content('current-version')) for image_info in result.get_child_by_name('attributes-list').get_children()] if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0 else []