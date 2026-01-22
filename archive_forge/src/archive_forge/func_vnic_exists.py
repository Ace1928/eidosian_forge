from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def vnic_exists(self):
    cmd = [self.module.get_bin_path('dladm', True)]
    cmd.append('show-vnic')
    cmd.append(self.name)
    rc, dummy, dummy = self.module.run_command(cmd)
    if rc == 0:
        return True
    else:
        return False