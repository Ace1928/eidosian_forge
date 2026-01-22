from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def eradicate_pod(module, array):
    """Eradicate Deleted Pod"""
    changed = True
    if not module.check_mode:
        if module.params['eradicate']:
            try:
                array.eradicate_pod(module.params['name'])
            except Exception:
                module.fail_json(msg='Eradication of pod {0} failed'.format(module.params['name']))
    module.exit_json(changed=changed)