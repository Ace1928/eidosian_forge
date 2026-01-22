from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_modify_actions(self, current):
    modify = self.na_helper.get_modified_attributes(current, self.parameters)
    if 'path' in modify:
        self.module.fail_json(msg='Error: mismatch on path values: desired: %s, received: %s' % (self.parameters['path'], current['path']))
    if 'acls' in modify:
        acl_actions = self.get_acl_actions_on_modify(modify, current)
        del modify['acls']
    else:
        acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
    if not any((acl_actions['patch-acls'], acl_actions['post-acls'], acl_actions['delete-acls'], modify)):
        self.na_helper.changed = False
    return (modify, acl_actions)