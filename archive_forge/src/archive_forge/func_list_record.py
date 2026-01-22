from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
def list_record(self, record):
    search = 'list %s' % record.replace(';', '&&').replace('set', 'where')
    cmd = [self.module.get_bin_path('ipwcli', True), '-user=%s' % self.user, '-password=%s' % self.password]
    rc, out, err = self.module.run_command(cmd, data=search)
    if 'Invalid username or password' in out:
        self.module.fail_json(msg='access denied at ipwcli login: Invalid username or password')
    if 'ARecord %s' % self.dnsname in out and rc == 0 or ('SRVRecord %s' % self.dnsname in out and rc == 0) or ('NAPTRRecord %s' % self.dnsname in out and rc == 0):
        return (True, rc, out, err)
    return (False, rc, out, err)