from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def find_user_account(self):
    searchStr = self.local_user_name
    exactMatch = True
    findUsers = True
    findGroups = False
    user_account = self.content.userDirectory.RetrieveUserGroups(None, searchStr, None, None, exactMatch, findUsers, findGroups)
    return user_account