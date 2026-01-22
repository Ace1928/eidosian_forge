from __future__ import absolute_import, division, print_function
import crypt
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
def prepare_modify_user_command(self):
    record = self.create_json_record()
    cmd = [self.module.get_bin_path('homectl', True)]
    cmd.append('update')
    cmd.append(self.name)
    cmd.append('--identity=-')
    if self.disksize and self.resize:
        cmd.append('--and-resize')
        cmd.append('true')
        self.result['changed'] = True
    return (cmd, record)