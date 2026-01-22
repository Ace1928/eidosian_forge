from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def enable_nfs(self):
    """
        enable nfs (online). If the NFS service was not explicitly created,
        this API will create one with default options.
        """
    nfs_enable = netapp_utils.zapi.NaElement.create_node_with_children('nfs-enable')
    try:
        self.server.invoke_successfully(nfs_enable, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error changing the service_state of nfs %s to %s: %s' % (self.parameters['vserver'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())