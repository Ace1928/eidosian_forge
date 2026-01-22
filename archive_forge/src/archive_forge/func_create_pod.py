from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def create_pod(module, array):
    """Create Pod"""
    changed = True
    if module.params['target']:
        module.fail_json(msg='Cannot clone non-existant pod.')
    if not module.check_mode:
        try:
            if module.params['failover']:
                array.create_pod(module.params['name'], failover_list=module.params['failover'])
            else:
                array.create_pod(module.params['name'])
        except Exception:
            module.fail_json(msg='Pod {0} creation failed.'.format(module.params['name']))
        if module.params['mediator'] != 'purestorage':
            try:
                array.set_pod(module.params['name'], mediator=module.params['mediator'])
            except Exception:
                module.warn('Failed to communicate with mediator {0}, using default value'.format(module.params['mediator']))
        if module.params['stretch']:
            current_array = array.get()['array_name']
            if module.params['stretch'] != current_array:
                try:
                    array.add_pod(module.params['name'], module.params['rrays'])
                except Exception:
                    module.fail_json(msg='Failed to stretch pod {0} to array {1}.'.format(module.params['name'], module.params['stretch']))
        if module.params['quota']:
            arrayv6 = get_array(module)
            res = arrayv6.patch_pods(names=[module.params['name']], pod=flasharray.PodPatch(quota_limit=human_to_bytes(module.params['quota'])))
            if res.status_code != 200:
                module.fail_json(msg='Failed to apply quota to pod {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)