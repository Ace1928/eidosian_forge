from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def copy_lun(self):
    """
        Copy LUN with requested path and vserver
        """
    if self.use_rest:
        return self.copy_lun_rest()
    lun_copy = netapp_utils.zapi.NaElement.create_node_with_children('lun-copy-start', **{'source-vserver': self.parameters['source_vserver']})
    path_obj = netapp_utils.zapi.NaElement('paths')
    pair = netapp_utils.zapi.NaElement('lun-path-pair')
    pair.add_new_child('destination-path', self.parameters['destination_path'])
    pair.add_new_child('source-path', self.parameters['source_path'])
    path_obj.add_child_elem(pair)
    lun_copy.add_child_elem(path_obj)
    try:
        self.server.invoke_successfully(lun_copy, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error copying lun from %s to  vserver %s: %s' % (self.parameters['source_vserver'], self.parameters['destination_vserver'], to_native(e)), exception=traceback.format_exc())