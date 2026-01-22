from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def kernel_persist_check():
    return module.params.get('kernel_params') or module.params.get('initrd_path') or (module.params.get('kernel_path') and (not module.params.get('cloud_init_persist')))