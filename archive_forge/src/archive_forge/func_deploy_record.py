from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
def deploy_record(self, record):
    stdin = 'create %s' % record
    cmd = [self.module.get_bin_path('ipwcli', True), '-user=%s' % self.user, '-password=%s' % self.password]
    rc, out, err = self.module.run_command(cmd, data=stdin)
    if 'Invalid username or password' in out:
        self.module.fail_json(msg='access denied at ipwcli login: Invalid username or password')
    if '1 object(s) created.' in out:
        return (rc, out, err)
    else:
        self.module.fail_json(msg='record creation failed', stderr=out)