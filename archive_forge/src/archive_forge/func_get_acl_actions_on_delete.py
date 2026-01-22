from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_acl_actions_on_delete(self, current):
    acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
    self.na_helper.changed = False
    if current.get('acls'):
        for acl in current['acls']:
            if not acl.get('inherited') and self.parameters.get('access_control') in (None, acl.get('access_control')):
                self.na_helper.changed = True
                acl_actions['delete-acls'].append(acl)
    return acl_actions