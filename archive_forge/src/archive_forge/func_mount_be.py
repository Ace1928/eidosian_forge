from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def mount_be(self):
    cmd = [self.module.get_bin_path('beadm'), 'mount', self.name]
    if self.mountpoint:
        cmd.append(self.mountpoint)
    return self.module.run_command(cmd)