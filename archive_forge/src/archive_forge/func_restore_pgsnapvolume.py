from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def restore_pgsnapvolume(module, array):
    """Restore a Protection Group Snapshot Volume"""
    changed = True
    if module.params['suffix'] == 'latest':
        latest_snapshot = list(array.get_protection_group_snapshots(names=[module.params['name']]).items)[-1].suffix
        module.params['suffix'] = latest_snapshot
    if ':' in module.params['name'] and '::' not in module.params['name']:
        if get_rpgsnapshot(module, array) is None:
            module.fail_json(msg='Selected restore snapshot {0} does not exist in the Protection Group'.format(module.params['restore']))
    elif get_pgroupvolume(module, array) is None:
        module.fail_json(msg='Selected restore volume {0} does not exist in the Protection Group'.format(module.params['restore']))
    source_volume = module.params['name'] + '.' + module.params['suffix'] + '.' + module.params['restore']
    if '::' in module.params['target']:
        target_pod_name = module.params['target'].split(':')[0]
        if '::' in module.params['name']:
            source_pod_name = module.params['name'].split(':')[0]
        else:
            source_pod_name = ''
        if source_pod_name != target_pod_name:
            if list(array.get_pods(names=[target_pod_name]).items)[0].array_count > 1:
                module.fail_json(msg='Volume cannot be restored to a stretched pod')
    if not module.check_mode:
        if LooseVersion(DEFAULT_API) <= LooseVersion(array.get_rest_version()):
            if module.params['add_to_pgs']:
                add_to_pgs = []
                for add_pg in range(0, len(module.params['add_to_pgs'])):
                    add_to_pgs.append(FixedReference(name=module.params['add_to_pgs'][add_pg]))
                res = array.post_volumes(names=[module.params['target']], volume=VolumePost(source=Reference(name=source_volume)), with_default_protection=module.params['with_default_protection'], add_to_protection_group_names=add_to_pgs)
            elif module.params['overwrite']:
                res = array.post_volumes(names=[module.params['target']], volume=VolumePost(source=Reference(name=source_volume)), overwrite=module.params['overwrite'])
            else:
                res = array.post_volumes(names=[module.params['target']], volume=VolumePost(source=Reference(name=source_volume)), with_default_protection=module.params['with_default_protection'])
        else:
            res = array.post_volumes(names=[module.params['target']], overwrite=module.params['overwrite'], volume=VolumePost(source=Reference(name=source_volume)))
        if res.status_code != 200:
            module.fail_json(msg='Failed to restore {0} from pgroup {1}. Error: {2}'.format(module.params['restore'], module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)