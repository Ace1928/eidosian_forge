from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils import six
from ansible_collections.community.general.plugins.module_utils.xenserver import (
@staticmethod
def get_cdrom_type(vm_cdrom_params):
    """Returns VM CD-ROM type."""
    if vm_cdrom_params['empty']:
        return 'none'
    else:
        return 'iso'