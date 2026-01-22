from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_package_delete(self):
    """
        Delete current cluster image package
        """
    if self.use_rest:
        return self.cluster_image_package_delete_rest()
    cluster_image_package_delete_info = netapp_utils.zapi.NaElement('cluster-image-package-delete')
    cluster_image_package_delete_info.add_new_child('package-version', self.parameters['package_version'])
    try:
        self.server.invoke_successfully(cluster_image_package_delete_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting cluster image package for %s: %s' % (self.parameters['package_version'], to_native(error)), exception=traceback.format_exc())