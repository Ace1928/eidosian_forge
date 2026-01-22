from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def recover_volume(module, array):
    """Recover Deleted Volume"""
    changed = True
    volfact = []
    if not module.check_mode:
        try:
            array.recover_volume(module.params['name'])
        except Exception:
            module.fail_json(msg='Recovery of volume {0} failed'.format(module.params['name']))
        volfact = array.get_volume(module.params['name'])
        volfact['page83_naa'] = PURE_OUI + volfact['serial'].lower()
        volfact['nvme_nguid'] = _create_nguid(volfact['serial'].lower())
    module.exit_json(changed=changed, volume=volfact)