from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_volume_clone(self):
    """
        Creates a new volume clone
        """
    if self.use_rest:
        return self.create_volume_clone_rest()
    clone_obj = netapp_utils.zapi.NaElement('volume-clone-create')
    clone_obj.add_new_child('parent-volume', self.parameters['parent_volume'])
    clone_obj.add_new_child('volume', self.parameters['name'])
    if self.parameters.get('qos_policy_group_name'):
        clone_obj.add_new_child('qos-policy-group-name', self.parameters['qos_policy_group_name'])
    if self.parameters.get('space_reserve'):
        clone_obj.add_new_child('space-reserve', self.parameters['space_reserve'])
    if self.parameters.get('parent_snapshot'):
        clone_obj.add_new_child('parent-snapshot', self.parameters['parent_snapshot'])
    if self.parameters.get('parent_vserver'):
        clone_obj.add_new_child('parent-vserver', self.parameters['parent_vserver'])
        clone_obj.add_new_child('vserver', self.parameters['vserver'])
    if self.parameters.get('volume_type'):
        clone_obj.add_new_child('volume-type', self.parameters['volume_type'])
    if self.parameters.get('junction_path'):
        clone_obj.add_new_child('junction-path', self.parameters['junction_path'])
    if self.parameters.get('uid'):
        clone_obj.add_new_child('uid', str(self.parameters['uid']))
        clone_obj.add_new_child('gid', str(self.parameters['gid']))
    try:
        self.create_server.invoke_successfully(clone_obj, True)
    except netapp_utils.zapi.NaApiError as exc:
        self.module.fail_json(msg='Error creating volume clone: %s: %s' % (self.parameters['name'], to_native(exc)))