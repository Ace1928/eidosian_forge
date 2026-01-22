from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
def nictag_exists(self):
    cmd = [self.nictagadm_bin, 'exists', self.name]
    rc, dummy, dummy = self.module.run_command(cmd)
    return rc == 0