from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def rename_endpoint(module, array):
    """Rename endpoint within a container, ie vgroup or local array"""
    changed = False
    volfact = []
    target_name = module.params['rename']
    if '/' in module.params['rename'] or '::' in module.params['rename']:
        module.fail_json(msg='Target endpoint cannot include a container name')
    if '/' in module.params['name']:
        vgroup_name = module.params['name'].split('/')[0]
        target_name = vgroup_name + '/' + module.params['rename']
    if get_target(target_name, array) or get_destroyed_endpoint(target_name, array):
        module.fail_json(msg='Target endpoint {0} already exists.'.format(target_name))
    else:
        try:
            changed = True
            if not module.check_mode:
                volfact = array.rename_volume(module.params['name'], target_name)
        except Exception:
            module.fail_json(msg='Rename endpoint {0} to {1} failed.'.format(module.params['name'], module.params['rename']))
    module.exit_json(changed=changed, volume=volfact)