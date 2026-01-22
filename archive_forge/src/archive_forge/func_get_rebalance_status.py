from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from distutils.version import LooseVersion
def get_rebalance_status(name):
    out = run_gluster(['volume', 'rebalance', name, 'status'], environ_update=dict(LANG='C', LC_ALL='C', LC_MESSAGES='C'))
    raw_out = out.split('\n')
    rebalance_status = []
    for line in raw_out:
        line = ' '.join(line.split())
        line_vals = line.split(' ')
        if line_vals[0].startswith('-') or line_vals[0].startswith('Node'):
            continue
        node_dict = {}
        if len(line_vals) == 1 or len(line_vals) == 4:
            continue
        node_dict['node'] = line_vals[0]
        node_dict['rebalanced_files'] = line_vals[1]
        node_dict['failures'] = line_vals[4]
        if 'in progress' in line:
            node_dict['status'] = line_vals[5] + line_vals[6]
            rebalance_status.append(node_dict)
        elif 'completed' in line:
            node_dict['status'] = line_vals[5]
            rebalance_status.append(node_dict)
    return rebalance_status