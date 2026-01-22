from __future__ import absolute_import, division, print_function
import crypt
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
def get_user_metadata(self):
    cmd = [self.module.get_bin_path('homectl', True)]
    cmd.append('inspect')
    cmd.append(self.name)
    cmd.append('-j')
    cmd.append('--no-pager')
    rc, stdout, stderr = self.module.run_command(cmd)
    return (rc, stdout, stderr)