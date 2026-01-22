from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def enable_interface(self):
    cmd = [self.module.get_bin_path('ipadm', True)]
    cmd.append('enable-if')
    cmd.append('-t')
    cmd.append(self.name)
    return self.module.run_command(cmd)