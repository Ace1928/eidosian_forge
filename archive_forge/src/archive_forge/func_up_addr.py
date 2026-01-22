from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def up_addr(self):
    cmd = [self.module.get_bin_path('ipadm')]
    cmd.append('up-addr')
    if self.temporary:
        cmd.append('-t')
    cmd.append(self.addrobj)
    return self.module.run_command(cmd)