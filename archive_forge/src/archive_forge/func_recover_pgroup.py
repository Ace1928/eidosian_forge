from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def recover_pgroup(module, array):
    """Recover deleted protection group"""
    changed = True
    if not module.check_mode:
        if ':' in module.params['name']:
            if '::' not in module.params['name']:
                try:
                    target = ''.join(module.params['target'])
                    array.recover_pgroup(module.params['name'], on=target)
                except Exception:
                    module.fail_json(msg='Recover pgroup {0} failed.'.format(module.params['name']))
            else:
                try:
                    array.recover_pgroup(module.params['name'])
                except Exception:
                    module.fail_json(msg='Recover pgroup {0} failed.'.format(module.params['name']))
        else:
            try:
                array.recover_pgroup(module.params['name'])
            except Exception:
                module.fail_json(msg='ecover pgroup {0} failed.'.format(module.params['name']))
    module.exit_json(changed=changed)