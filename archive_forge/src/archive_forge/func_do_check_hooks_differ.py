from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def do_check_hooks_differ(current, desired):
    if 'check_hooks' not in desired:
        return False
    current = utils.single_item_dicts_to_dict(current.get('check_hooks') or [])
    current = dict(((k, set(v)) for k, v in current.items()))
    desired = utils.single_item_dicts_to_dict(desired['check_hooks'])
    desired = dict(((k, set(v)) for k, v in desired.items()))
    return current != desired