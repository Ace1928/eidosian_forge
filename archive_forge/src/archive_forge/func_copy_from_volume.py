from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def copy_from_volume(module, array):
    """Create Volume Clone"""
    volfact = []
    changed = False
    tgt = get_target(module, array)
    api_version = array._list_available_rest_versions()
    arrayv6 = get_array(module)
    if tgt is None:
        changed = True
        if not module.check_mode:
            if DEFAULT_API_VERSION in api_version:
                if module.params['add_to_pgs']:
                    add_to_pgs = []
                    for add_pg in range(0, len(module.params['add_to_pgs'])):
                        add_to_pgs.append(flasharray.FixedReference(name=module.params['add_to_pgs'][add_pg]))
                    res = arrayv6.post_volumes(with_default_protection=module.params['with_default_protection'], add_to_protection_groups=add_to_pgs, names=[module.params['target']], volume=flasharray.VolumePost(source=flasharray.Reference(name=module.params['name'])))
                else:
                    res = arrayv6.post_volumes(with_default_protection=module.params['with_default_protection'], names=[module.params['target']], volume=flasharray.VolumePost(source=flasharray.Reference(name=module.params['name'])))
                if res.status_code != 200:
                    module.fail_json(msg='Failed to copy volume {0} to {1}. Error: {2}'.format(module.params['name'], module.params['target'], res.errors[0].message))
                vol_data = list(res.items)
                volfact = {'size': vol_data[0].provisioned, 'serial': vol_data[0].serial, 'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(vol_data[0].created / 1000)), 'page83_naa': PURE_OUI + vol_data[0].serial.lower(), 'nvme_nguid': _create_nguid(vol_data[0].serial.lower())}
            else:
                try:
                    volfact = array.copy_volume(module.params['name'], module.params['target'])
                    volfact['page83_naa'] = PURE_OUI + volfact['serial'].lower()
                    volfact['nvme_nguid'] = _create_nguid(volfact['serial'].lower())
                    changed = True
                except Exception:
                    module.fail_json(msg='Copy volume {0} to volume {1} failed.'.format(module.params['name'], module.params['target']))
    elif tgt is not None and module.params['overwrite']:
        changed = True
        if not module.check_mode:
            if DEFAULT_API_VERSION not in api_version:
                try:
                    volfact = array.copy_volume(module.params['name'], module.params['target'], overwrite=module.params['overwrite'])
                    volfact['page83_naa'] = PURE_OUI + volfact['serial'].lower()
                    volfact['nvme_nguid'] = _create_nguid(volfact['serial'].lower())
                    changed = True
                except Exception:
                    module.fail_json(msg='Copy volume {0} to volume {1} failed.'.format(module.params['name'], module.params['target']))
            else:
                res = arrayv6.post_volumes(overwrite=module.params['overwrite'], names=[module.params['target']], volume=flasharray.VolumePost(source=flasharray.Reference(name=module.params['name'])))
                if res.status_code != 200:
                    module.fail_json(msg='Failed to copy volume {0} to {1}. Error: {2}'.format(module.params['name'], module.params['target'], res.errors[0].message))
                vol_data = list(res.items)
                volfact = {'size': vol_data[0].provisioned, 'serial': vol_data[0].serial, 'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(vol_data[0].created / 1000)), 'page83_naa': PURE_OUI + vol_data[0].serial.lower(), 'nvme_nguid': _create_nguid(vol_data[0].serial.lower())}
    module.exit_json(changed=changed, volume=volfact)