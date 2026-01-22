from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_cluster_identity(self, ignore_error=True):
    """ get cluster information, but the cluster may not exist yet
            return:
                None if the cluster cannot be reached
                a dictionary of attributes
        """
    if self.use_rest:
        return self.get_cluster_identity_rest()
    zapi = netapp_utils.zapi.NaElement('cluster-identity-get')
    try:
        result = self.server.invoke_successfully(zapi, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if ignore_error:
            return None
        self.module.fail_json(msg='Error fetching cluster identity info: %s' % to_native(error), exception=traceback.format_exc())
    cluster_identity = {}
    if result.get_child_by_name('attributes'):
        identity_info = result.get_child_by_name('attributes').get_child_by_name('cluster-identity-info')
        if identity_info:
            cluster_identity['cluster_contact'] = identity_info.get_child_content('cluster-contact')
            cluster_identity['cluster_location'] = identity_info.get_child_content('cluster-location')
            cluster_identity['cluster_name'] = identity_info.get_child_content('cluster-name')
        return cluster_identity
    return None