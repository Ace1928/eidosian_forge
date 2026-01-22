from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from copy import deepcopy
def check_hba_name(self):
    if self.existing_system_iscsi_config['vmhba_name'] != self.vmhba_name:
        self.module.fail_json(msg='%s is not an iSCSI device.' % self.vmhba_name)