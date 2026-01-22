from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
def populate_role_list(self):
    user_friendly_role_names = {'Admin': ['Administrator'], 'ReadOnly': ['Read-Only'], 'com.vmware.Content.Admin': ['Content library administrator (sample)', 'Content library administrator'], 'NoCryptoAdmin': ['No cryptography administrator'], 'NoAccess': ['No access'], 'VirtualMachinePowerUser': ['Virtual machine power user (sample)', 'Virtual machine power user'], 'VirtualMachineUser': ['Virtual machine user (sample)', 'Virtual machine user'], 'ResourcePoolAdministrator': ['Resource pool administrator (sample)', 'Resource pool administrator'], 'VMwareConsolidatedBackupUser': ['VMware Consolidated Backup user (sample)', 'VMware Consolidated Backup user'], 'DatastoreConsumer': ['Datastore consumer (sample)', 'Datastore consumer'], 'NetworkConsumer': ['Network administrator (sample)', 'Network administrator'], 'VirtualMachineConsoleUser': ['Virtual Machine console user'], 'InventoryService.Tagging.TaggingAdmin': ['Tagging Admin']}
    for role in self.auth_manager.roleList:
        self.role_list[role.name] = role
        if user_friendly_role_names.get(role.name):
            for role_name in user_friendly_role_names[role.name]:
                self.role_list[role_name] = role