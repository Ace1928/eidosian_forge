from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def update_flow(self):
    cmd = [self.module.get_bin_path('flowadm')]
    cmd.append('set-flowprop')
    if self.maxbw and self._needs_updating['maxbw']:
        cmd.append('-p')
        cmd.append('maxbw=' + self.maxbw)
    if self.priority and self._needs_updating['priority']:
        cmd.append('-p')
        cmd.append('priority=' + self.priority)
    if self.temporary:
        cmd.append('-t')
    cmd.append(self.name)
    return self.module.run_command(cmd)