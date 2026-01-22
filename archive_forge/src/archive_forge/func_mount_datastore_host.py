from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
def mount_datastore_host(self):
    if self.datastore_type == 'nfs' or self.datastore_type == 'nfs41':
        self.mount_nfs_datastore_host()
    if self.datastore_type == 'vmfs':
        self.mount_vmfs_datastore_host()
    if self.datastore_type == 'vvol':
        self.mount_vvol_datastore_host()