from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def check_pool_exclusiveness(self, mastervol, slavevol):
    rc, output, err = self.module.run_command('gluster pool list')
    peers_in_cluster = [line.split('\t')[1].strip() for line in filter(None, output.split('\n')[1:])]
    val_group = re.search('(.*):(.*)', slavevol)
    if not val_group:
        self.module.fail_json(msg='Slave volume in Unknown format. Correct format: <hostname>:<volume name>')
    if val_group.group(1) in peers_in_cluster:
        self.module.fail_json(msg='slave volume is in the trusted storage pool of master')
    self.user = 'root' if self.module.params['georepuser'] is None else self.module.params['georepuser']
    return self.user + '@' + val_group.group(1) + '::' + val_group.group(2)