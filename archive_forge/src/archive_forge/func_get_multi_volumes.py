from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_multi_volumes(module, destroyed=False):
    """Return True is all volumes exist or None"""
    names = []
    array = get_array(module)
    for vol_num in range(module.params['start'], module.params['count'] + module.params['start']):
        names.append(module.params['name'] + str(vol_num).zfill(module.params['digits']) + module.params['suffix'])
    return bool(array.get_volumes(names=names, destroyed=destroyed).status_code == 200)