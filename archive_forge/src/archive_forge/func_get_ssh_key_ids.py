from __future__ import absolute_import, division, print_function
import base64
from .vultr_v2 import AnsibleVultr
def get_ssh_key_ids(self):
    ssh_key_names = list(self.module.params['ssh_keys'])
    ssh_keys = self.query_list(path='/ssh-keys', result_key='ssh_keys')
    ssh_key_ids = list()
    for ssh_key in ssh_keys:
        if ssh_key['name'] in ssh_key_names:
            ssh_key_ids.append(ssh_key['id'])
            ssh_key_names.remove(ssh_key['name'])
    if ssh_key_names:
        self.module.fail_json(msg='SSH key names not found: %s' % ', '.join(ssh_key_names))
    return ssh_key_ids