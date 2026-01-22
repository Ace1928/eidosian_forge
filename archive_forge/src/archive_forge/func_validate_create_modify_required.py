from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_create_modify_required(self, current, modify):
    new_current = self.get_role()
    new_cd_action = self.na_helper.get_cd_action(new_current, self.parameters)
    new_modify = None if new_cd_action else self.na_helper.get_modified_attributes(new_current, self.parameters)
    msg = ''
    if current is None and new_modify:
        msg = 'Create operation also affected additional related commands: %s' % new_current['privileges']
    elif modify and new_cd_action == 'create':
        msg = "Create role is required, desired is: %s but it's a subset of relevant commands/command directory configured in current: %s,\n                     deleting one of the commands will remove all the commands in the relevant group" % (self.parameters['privileges'], current['privileges'])
    elif modify and new_modify:
        msg = 'modify is required, desired: %s and new current: %s' % (self.parameters['privileges'], new_current['privileges'])
    if msg:
        self.module.warn(msg)