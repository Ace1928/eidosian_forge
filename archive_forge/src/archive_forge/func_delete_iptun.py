from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def delete_iptun(self):
    cmd = [self.dladm_bin]
    cmd.append('delete-iptun')
    if self.temporary:
        cmd.append('-t')
    cmd.append(self.name)
    return self.module.run_command(cmd)