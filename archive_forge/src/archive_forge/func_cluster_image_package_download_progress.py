from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_package_download_progress(self):
    """
        Get current cluster image package download progress
        :return: Dictionary of cluster image download progress if query successful, else return None
        """
    cluster_image_package_download_progress_info = netapp_utils.zapi.NaElement('cluster-image-get-download-progress')
    try:
        result = self.server.invoke_successfully(cluster_image_package_download_progress_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching cluster image package download progress for %s: %s' % (self.parameters['package_url'], to_native(error)), exception=traceback.format_exc())
    cluster_download_progress_info = {}
    if result.get_child_by_name('progress-status'):
        cluster_download_progress_info['progress_status'] = result.get_child_content('progress-status')
        cluster_download_progress_info['progress_details'] = result.get_child_content('progress-details')
        cluster_download_progress_info['failure_reason'] = result.get_child_content('failure-reason')
        return cluster_download_progress_info
    return None