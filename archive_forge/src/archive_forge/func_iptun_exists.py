from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def iptun_exists(self):
    cmd = [self.dladm_bin]
    cmd.append('show-iptun')
    cmd.append(self.name)
    rc, dummy, dummy = self.module.run_command(cmd)
    if rc == 0:
        return True
    else:
        return False