from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
def is_dhcp(self):
    cmd = [self.module.get_bin_path('ipadm')]
    cmd.append('show-addr')
    cmd.append('-p')
    cmd.append('-o')
    cmd.append('type')
    cmd.append(self.addrobj)
    rc, out, err = self.module.run_command(cmd)
    if rc == 0:
        if out.rstrip() != 'dhcp':
            return False
        return True
    else:
        self.module.fail_json(msg='Wrong addrtype %s for addrobj "%s": %s' % (out, self.addrobj, err), rc=rc, stderr=err)