from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def determine_cwd(scope, params):
    if scope == 'local':
        return params['repo']
    elif params['list_all'] and params['repo']:
        return params['repo']
    else:
        return '/'