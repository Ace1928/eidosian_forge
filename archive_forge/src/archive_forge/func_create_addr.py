from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def create_addr(self):
    cmd = [self.module.get_bin_path('ipadm')]
    cmd.append('create-addr')
    cmd.append('-T')
    cmd.append(self.addrtype)
    if self.temporary:
        cmd.append('-t')
    if self.addrtype == 'static':
        cmd.append('-a')
        cmd.append(self.address)
    if self.addrtype == 'dhcp' and self.wait:
        cmd.append('-w')
        cmd.append(self.wait)
    cmd.append(self.addrobj)
    return self.module.run_command(cmd)