from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def interface_is_disabled(self):
    cmd = [self.module.get_bin_path('ipadm', True)]
    cmd.append('show-if')
    cmd.append('-o')
    cmd.append('state')
    cmd.append(self.name)
    rc, out, err = self.module.run_command(cmd)
    if rc != 0:
        self.module.fail_json(name=self.name, rc=rc, msg=err)
    return 'disabled' in out