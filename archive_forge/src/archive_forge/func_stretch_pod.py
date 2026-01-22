from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def stretch_pod(module, array):
    """Stretch/unstretch Pod configuration"""
    changed = False
    current_config = array.get_pod(module.params['name'], failover_preference=True)
    if module.params['stretch']:
        current_arrays = []
        for arr in range(0, len(current_config['arrays'])):
            current_arrays.append(current_config['arrays'][arr]['name'])
        if module.params['stretch'] not in current_arrays and module.params['state'] == 'present':
            changed = True
            if not module.check_mode:
                try:
                    array.add_pod(module.params['name'], module.params['stretch'])
                except Exception:
                    module.fail_json(msg='Failed to stretch pod {0} to array {1}.'.format(module.params['name'], module.params['stretch']))
        if module.params['stretch'] in current_arrays and module.params['state'] == 'absent':
            changed = True
            if not module.check_mode:
                try:
                    array.remove_pod(module.params['name'], module.params['stretch'])
                except Exception:
                    module.fail_json(msg='Failed to unstretch pod {0} from array {1}.'.format(module.params['name'], module.params['stretch']))
    module.exit_json(changed=changed)