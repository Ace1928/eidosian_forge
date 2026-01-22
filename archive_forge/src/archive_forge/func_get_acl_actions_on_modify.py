from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_acl_actions_on_modify(self, modify, current):
    acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
    if not self.has_acls(current):
        acl_actions['post-acls'] = modify['acls']
        return acl_actions
    for acl in modify['acls']:
        current_acl = self.match_acl_with_acls(acl, current['acls'])
        if current_acl:
            if self.is_modify_acl_required(acl, current_acl):
                acl_actions['patch-acls'].append(acl)
        else:
            acl_actions['post-acls'].append(acl)
    for acl in current['acls']:
        desired_acl = self.match_acl_with_acls(acl, self.parameters['acls'])
        if not desired_acl and (not acl.get('inherited')) and (self.parameters.get('access_control') in (None, acl.get('access_control'))):
            acl_actions['delete-acls'].append(acl)
    return acl_actions