from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule, is_executable
def get_matched_processes():
    matched = []
    rc, out, err = run_supervisorctl('status')
    for line in out.splitlines():
        fields = [field for field in line.split(' ') if field != '']
        process_name = fields[0]
        status = fields[1]
        if is_group:
            if ':' in process_name:
                group = process_name.split(':')[0]
                if group != name:
                    continue
            else:
                continue
        elif process_name != name and name != 'all':
            continue
        matched.append((process_name, status))
    return matched