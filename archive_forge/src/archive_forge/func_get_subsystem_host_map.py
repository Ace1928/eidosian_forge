from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_subsystem_host_map(self, type):
    """
        Get current subsystem host details
        :return: list if host exists, None otherwise
        """
    if type == 'hosts':
        zapi_get, zapi_info, zapi_type = ('nvme-subsystem-host-get-iter', 'nvme-target-subsystem-host-info', 'host-nqn')
    elif type == 'paths':
        zapi_get, zapi_info, zapi_type = ('nvme-subsystem-map-get-iter', 'nvme-target-subsystem-map-info', 'path')
    result = self.get_zapi_info(zapi_get, zapi_info, zapi_type)
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        attrs_list = result.get_child_by_name('attributes-list')
        return_list = [item[zapi_type] for item in attrs_list.get_children()]
        return {type: return_list}
    return None