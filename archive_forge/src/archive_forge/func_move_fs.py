from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def move_fs(module, array):
    """Move filesystem between pods or local array"""
    changed = False
    target_exists = False
    pod_name = ''
    fs_name = module.params['name']
    if '::' in module.params['name']:
        fs_name = module.params['name'].split('::')[1]
        pod_name = module.params['name'].split('::')[0]
    if module.params['move'] == 'local':
        target_location = ''
        if '::' not in module.params['name']:
            module.fail_json(msg='Source and destination [local] cannot be the same.')
        try:
            target_exists = list(array.get_file_systems(names=[fs_name]).items)[0]
        except Exception:
            target_exists = False
        if target_exists:
            module.fail_json(msg='Target filesystem {0} already exists'.format(fs_name))
    else:
        try:
            pod = list(array.get_pods(names=[module.params['move']]).items)[0]
            if len(pod.arrays) > 1:
                module.fail_json(msg='Filesystem cannot be moved into a stretched pod')
            if pod.link_target_count != 0:
                module.fail_json(msg='Filesystem cannot be moved into a linked source pod')
            if pod.promotion_status == 'demoted':
                module.fail_json(msg='Volume cannot be moved into a demoted pod')
        except Exception:
            module.fail_json(msg='Failed to move filesystem. Pod {0} does not exist'.format(pod_name))
        if '::' in module.params['name']:
            pod = list(array.get_pods(names=[module.params['move']]).items)[0]
            if len(pod.arrays) > 1:
                module.fail_json(msg='Filesystem cannot be moved out of a stretched pod')
            if pod.linked_target_count != 0:
                module.fail_json(msg='Filesystem cannot be moved out of a linked source pod')
            if pod.promotion_status == 'demoted':
                module.fail_json(msg='Volume cannot be moved out of a demoted pod')
        target_location = module.params['move']
    changed = True
    if not module.check_mode:
        file_system = flasharray.FileSystemPatch(pod=flasharray.Reference(name=target_location))
        move_res = array.patch_file_systems(names=[module.params['name']], file_system=file_system)
        if move_res.status_code != 200:
            module.fail_json(msg='Move of filesystem {0} failed. Error: {1}'.format(module.params['name'], move_res.errors[0].message))
    module.exit_json(changed=changed)