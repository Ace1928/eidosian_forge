from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def determine_scope(params):
    if params['scope']:
        return params['scope']
    elif params['list_all']:
        return ''
    else:
        return 'system'