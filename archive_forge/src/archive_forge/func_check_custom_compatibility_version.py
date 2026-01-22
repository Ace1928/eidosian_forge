from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def check_custom_compatibility_version():
    if self.param('custom_compatibility_version') is not None:
        return self._get_minor(self.param('custom_compatibility_version')) == self._get_minor(entity.custom_compatibility_version) and self._get_major(self.param('custom_compatibility_version')) == self._get_major(entity.custom_compatibility_version)
    return True