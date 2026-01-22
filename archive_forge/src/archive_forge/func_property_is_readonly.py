from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def property_is_readonly(self):
    cmd = [self.dladm_bin]
    cmd.append('show-linkprop')
    cmd.append('-c')
    cmd.append('-o')
    cmd.append('perm')
    cmd.append('-p')
    cmd.append(self.property)
    cmd.append(self.link)
    rc, out, dummy = self.module.run_command(cmd)
    out = out.rstrip()
    if rc == 0 and out == 'r-':
        return True
    else:
        return False